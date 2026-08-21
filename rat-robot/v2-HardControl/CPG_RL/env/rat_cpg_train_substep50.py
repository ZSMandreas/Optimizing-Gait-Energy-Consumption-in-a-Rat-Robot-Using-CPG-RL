"""
rat_cpg_train_substep50.py

Train PPO on RatCPGEnv (substep50): 每 env step 执行 K=50 次 mj_step；
CPG 相位每 sim step 更新，CPG 参数 (f, mu) 每 env step 更新。
步态频率 env ACTION_RANGES["f"]=(0.3, 1) Hz，mu=(0.2, 1.0)；训练初始 CPG 取范围中点。

Rollout：每次更新 41 env steps × n_envs（每 env step = 50 sim steps → 2050 sim steps/episode）。
  --n-steps 41
  --max-episode-steps 2050（sim steps = 41 env steps）
  --batch-size 41（整除 41×n_envs）
  --save-freq 10000（env steps）
  TensorBoard：step=env steps，并记录 rollout/sim_steps_total=step×50。

Usage:
  python rat_cpg_train_substep50.py --log-dir logs/rat_cpg_substep50
  python rat_cpg_train_substep50.py --timesteps 400000
"""

import argparse
import os
import json
from collections import deque

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from rat_cpg_env_energy_substep50 import RatCPGEnv


# -----------------------------
# 做法B: 仅每 repeat_interval 个 env step 向策略要新动作，其余步返回缓存动作，与 env 的 action_update_interval 一致
# -----------------------------
class ActionRepeatModelWrapper:
    """使 rollout 中存储的动作与实际执行一致：每 repeat_interval 步才调用内部策略，中间步返回上一步动作。"""

    def __init__(self, model, repeat_interval: int):
        self.model = model
        self.repeat_interval = max(1, int(repeat_interval))
        self._step_count = 0
        self._cached_actions = None
        self._cached_state = None

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        self._step_count += 1
        if (self._step_count - 1) % self.repeat_interval == 0 or self._cached_actions is None:
            self._cached_actions, self._cached_state = self.model.predict(
                obs, state=state, episode_start=episode_start, deterministic=deterministic
            )
        return self._cached_actions, self._cached_state

    def __getattr__(self, name):
        return getattr(self.model, name)


# -----------------------------
# TensorBoard: episode-aligned rollout means
# -----------------------------

class EpisodeAlignedTBCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.buf_total = deque(maxlen=window_size)
        self.buf_r_vel = deque(maxlen=window_size)
        self.buf_r_pose = deque(maxlen=window_size)
        self.buf_r_xvel = deque(maxlen=window_size)
        self.buf_r_energy = deque(maxlen=window_size)
        self.buf_r_tau = deque(maxlen=window_size)
        # ET-aware diagnostics (optional; present only in ET-aware env info)
        self.buf_r_consistency = deque(maxlen=window_size)
        self.buf_r_execution = deque(maxlen=window_size)
        self.buf_delta_q_l2 = deque(maxlen=window_size)
        self.buf_delta_q_abs_mean = deque(maxlen=window_size)
        self.buf_r_sync = deque(maxlen=window_size)

    @staticmethod
    def _mean(buf):
        return float(sum(buf) / len(buf)) if len(buf) > 0 else 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if isinstance(infos, list):
            for info in infos:
                if not isinstance(info, dict):
                    continue
                comps = info.get("episode_components", None)
                if isinstance(comps, dict):
                    self.buf_total.append(float(comps.get("ep_total_reward", 0.0)))
                    self.buf_r_vel.append(float(comps.get("ep_r_vel", 0.0)))
                    self.buf_r_pose.append(float(comps.get("ep_r_pose", 0.0)))
                    self.buf_r_xvel.append(float(comps.get("ep_r_xvel", 0.0)))
                    self.buf_r_energy.append(float(comps.get("ep_r_energy", 0.0)))
                    self.buf_r_tau.append(float(comps.get("ep_r_tau", 0.0)))
                if "r_consistency" in info:
                    self.buf_r_consistency.append(float(info.get("r_consistency", 0.0)))
                if "r_execution" in info:
                    self.buf_r_execution.append(float(info.get("r_execution", 0.0)))
                if "delta_q_l2" in info:
                    self.buf_delta_q_l2.append(float(info.get("delta_q_l2", 0.0)))
                if "delta_q_abs_mean" in info:
                    self.buf_delta_q_abs_mean.append(float(info.get("delta_q_abs_mean", 0.0)))
                if "r_sync" in info:
                    self.buf_r_sync.append(float(info.get("r_sync", 0.0)))

        self.logger.record("rollout/ep_total_reward_mean_aligned", self._mean(self.buf_total))
        self.logger.record("rollout/ep_r_vel_mean_aligned", self._mean(self.buf_r_vel))
        self.logger.record("rollout/ep_r_pose_mean_aligned", self._mean(self.buf_r_pose))
        self.logger.record("rollout/ep_r_xvel_mean_aligned", self._mean(self.buf_r_xvel))
        self.logger.record("rollout/ep_r_energy_mean_aligned", self._mean(self.buf_r_energy))
        self.logger.record("rollout/ep_r_tau_mean_aligned", self._mean(self.buf_r_tau))
        self.logger.record("rollout/r_consistency_mean_aligned", self._mean(self.buf_r_consistency))
        self.logger.record("rollout/r_execution_mean_aligned", self._mean(self.buf_r_execution))
        self.logger.record("rollout/delta_q_l2_mean_aligned", self._mean(self.buf_delta_q_l2))
        self.logger.record("rollout/delta_q_abs_mean_aligned", self._mean(self.buf_delta_q_abs_mean))
        self.logger.record("rollout/r_sync_mean_aligned", self._mean(self.buf_r_sync))
        # 对应 substep50：总 sim steps = env steps × 50
        total_env_steps = getattr(self.model, "num_timesteps", 0) or getattr(self, "num_timesteps", 0)
        self.logger.record("rollout/sim_steps_total", total_env_steps * 50)
        return True


# -----------------------------
# Optional MuJoCo viewer callback
# -----------------------------

class MujocoRenderCallback(BaseCallback):
    def __init__(self, vec_env, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.viewer = None
        self._viewer_ok = False

    def _try_init_viewer(self):
        try:
            import mujoco.viewer  # noqa: F401
            env0 = self.vec_env.envs[0]
            unwrapped = env0.unwrapped
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(unwrapped.model, unwrapped.data)
            self._viewer_ok = True
        except Exception as e:
            self._viewer_ok = False
            if self.verbose > 0:
                print(f"[RenderCallback] Viewer unavailable: {e}")

    def _on_training_start(self) -> None:
        self._try_init_viewer()

    def _on_step(self) -> bool:
        if self._viewer_ok and self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass
        return True

    def _on_training_end(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass


# -----------------------------
# CLI (defaults for substep50 env)
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on RatCPGEnv (substep50)")

    p.add_argument("--timesteps", type=int, default=1000_000, help="Total env steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--log-dir", type=str, default="logs/rat_cpg_substep50", help="Log/model dir")
    p.add_argument("--tb-log-dir", type=str, default="", help="TensorBoard dir (default: <log-dir>/tb)")

    p.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint save frequency (env steps)")
    p.add_argument("--render", action="store_true", help="Visualize with MuJoCo viewer (n_envs=1)")
    p.add_argument("--print-sim-step-joint-angles", action="store_true", help="Print target/actual joint angles every sim step")
    p.add_argument("--print-sim-step-torque", action="store_true", help="Print motor torque every sim step")

    # Env (substep50; 步态 f 在 env 中为 (0.3, 1) Hz)
    p.add_argument("--model-path", type=str, default="/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml", help="MuJoCo XML path")
    p.add_argument("--gait-template", type=str, default="trot", choices=["trot", "pace", "bound", "walk"], help="Fixed gait phase template")
    p.add_argument("--target-speed", type=float, default=0.12, help="Target forward speed (along -Y), m/s")
    p.add_argument("--action-update-interval", type=int, default=1, help="CPG update interval (env steps); 1 = every 50 sim steps")
    p.add_argument("--smoothing-alpha", type=float, default=1, help="EMA smoothing for CPG params")
    p.add_argument("--max-episode-steps", type=int, default=4100, help="Episode cap (sim steps); 2050 = 41 env steps × 50 substeps")
    p.add_argument("--n-substeps", type=int, default=50, help="Substeps per env step (K)")
    p.add_argument("--log-interval", type=int, default=41, help="Logging interval (env steps); 41 = 1 rollout")
    p.add_argument("--randomize-cpg-params", action="store_true", help="Randomize CPG f/mu at reset (env randomize_cpg_params)")
    p.add_argument("--reset-f-sigma", type=float, default=0.05, help="Std (Hz) for initial f perturbation when --randomize-cpg-params (match 0.3–1 Hz range)")
    p.add_argument("--use-phase-for-contact", action="store_true", help="Use CPG phase for contact/stance (φ≥π), align with sim_test; else use sensordata")
    p.add_argument("--fixed-cpg", action="store_true", help="Ignore action, always use f4=0.5 Hz and mu4=1.0 (same as sim_test); for verifying env forward motion")
    p.add_argument("--use-energy-tank", action="store_true", help="Use energy tank (joint-space) between IK and ctrl; obs +2 dims (e_tank_norm, alpha_tank)")

    p.add_argument("--vel-weight", type=float, default=1.5, help="Velocity tracking weight")
    p.add_argument("--pose-weight", type=float, default=0.1, help="Stability penalty weight")
    p.add_argument("--smooth-weight", type=float, default=1e-3, help="Action smoothness weight")
    p.add_argument("--energy-weight", type=float, default=0.05, help="Energy weight (if env uses it)")
    p.add_argument("--x-weight", type=float, default=0.2, help="X deviation penalty weight")
    p.add_argument("--ik-fail-weight", type=float, default=0.2, help="IK failure penalty weight")

    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--n-steps", type=int, default=82, help="Rollout env steps per env per update (41×n_envs; 41×50=2050 sim steps)")
    p.add_argument("--batch-size", type=int, default=82, help="Minibatch size (must divide n_steps×n_envs=164)")
    p.add_argument("--n-epochs", type=int, default=10, help="Epochs per update")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    tb_log_dir = args.tb_log_dir if args.tb_log_dir else os.path.join(args.log_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)

    with open(os.path.join(args.log_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    def make_env() -> gym.Env:
        env = RatCPGEnv(
            model_path=args.model_path,
            gait_template=args.gait_template,
            target_speed=args.target_speed,
            action_update_interval=args.action_update_interval,
            smoothing_alpha=args.smoothing_alpha,
            max_episode_steps=args.max_episode_steps,
            n_substeps=args.n_substeps,
            vel_weight=args.vel_weight,
            pose_weight=args.pose_weight,
            smooth_weight=args.smooth_weight,
            energy_weight=args.energy_weight,
            x_weight=args.x_weight,
            ik_fail_weight=args.ik_fail_weight,
            randomize_cpg_params=args.randomize_cpg_params,
            reset_f_sigma=args.reset_f_sigma,
            use_phase_for_contact=args.use_phase_for_contact,
            fixed_cpg=args.fixed_cpg,
            use_energy_tank=args.use_energy_tank,
            render_mode="human" if args.render else None,
            print_sim_step_joint_angles=args.print_sim_step_joint_angles,
            print_sim_step_torque=args.print_sim_step_torque,
        )
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env, make_env, make_env, make_env])
    n_envs = env.num_envs

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_freq_calls = max(args.save_freq // n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=ckpt_dir,
        name_prefix="rat_cpg_ppo_substep50",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    tb_info_cb = EpisodeAlignedTBCallback(window_size=100)
    callbacks = [checkpoint_cb, tb_info_cb]
    if args.render:
        callbacks.append(MujocoRenderCallback(env, verbose=1))
    callback = CallbackList(callbacks)

    # TensorBoard：使用默认 logger，learn() 会创建 tb_log_dir/tb_log_name/run_id 并写入
    model = PPO(
        "MlpPolicy",
        env,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log_dir,
    )

    if getattr(args, "action_update_interval", 1) > 1:
        model = ActionRepeatModelWrapper(model, args.action_update_interval)

    # 训练前打印 obs 中各状态名称及索引
    unwrapped = env.envs[0].unwrapped
    if hasattr(unwrapped, "get_obs_layout"):
        print("\n[Observation layout] obs 中用到的状态及其索引:")
        for name, idx_str in unwrapped.get_obs_layout():
            print(f"  obs[{idx_str}]: {name}")
        print(f"  total dim: {unwrapped.observation_space.shape[0]}\n")

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name="rat_cpg_ppo_substep50",
        callback=callback,
    )

    final_path = os.path.join(args.log_dir, "rat_cpg_ppo_substep50.zip")
    # 若包了 ActionRepeatModelWrapper，save 会转发到内部 PPO，保存的是实际策略
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")
    print(f"Checkpoints saved to {ckpt_dir} every {args.save_freq} env steps.")


if __name__ == "__main__":
    main()
