"""
rat_cpg_eval_substep50.py

Evaluate PPO on RatCPGEnv (substep50): 每 env step 执行 K=50 次 mj_step；
CPG 相位每 sim step 更新，CPG 参数每 env step 更新。与 train 一致：f=(0.3,1) Hz，mu=(0.2,1.0)。
默认 max_episode_steps=2050 sim steps = 41 env steps（= 1 rollout）；--max-steps 为 cap 的 env steps。
做法B：当 --action-update-interval > 1 时，仅每 N 个 env step 调用一次策略，中间步复用上一步动作，与训练一致。

Usage:
  python rat_cpg_eval_substep50.py --model logs/rat_cpg_substep50/rat_cpg_ppo_substep50.zip --episodes 5
  python rat_cpg_eval_substep50.py --model path/to/model.zip --episodes 2 --render --save-csv
  python rat_cpg_eval_substep50.py --model path/to/model.zip --max-steps 41
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from rat_cpg_env_energy_substep50 import RatCPGEnv


class ObsPadWrapper35to43(gym.ObservationWrapper):
    """Pad 35-dim obs to 43-dim for old checkpoints (insert 8 zeros as joint_vel after joint_pos)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32
        )

    def observation(self, obs):
        # 35: base_height(1)+lin_vel(3)+ang_vel(3)+joint_pos(8)+contact(4)+phase(8)+f4(4)+mu4(4)
        # 43: same but insert joint_vel(8) after joint_pos -> after index 15
        obs = np.asarray(obs, dtype=np.float32).ravel()
        if obs.size != 35:
            return obs
        pad = np.zeros(8, dtype=np.float32)
        return np.concatenate([obs[:15], pad, obs[15:35]], axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO on RatCPGEnv (substep50)")

    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip (final or checkpoint)")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    p.add_argument("--max-steps", type=int, default=82, help="Max env steps per episode (cap); 41 = 1 train rollout; use with --max-episode-steps for long eval")
    p.add_argument("--render", action="store_true", help="Visualize with MuJoCo viewer")
    p.add_argument("--seed", type=int, default=None, help="Random seed for env (reproducible eval); set to same value for same initial CPG phase etc.")

    # Env (substep50 defaults; match rat_cpg_train_substep50.py)
    p.add_argument("--model-path", type=str, default="/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml", help="MuJoCo XML path")
    p.add_argument("--gait-template", type=str, default="trot", choices=["trot", "pace", "bound", "walk"], help="Fixed gait phase template (must match training)")
    p.add_argument("--target-speed", type=float, default=0.12, help="Target forward speed (along -Y), m/s (match train)")
    p.add_argument("--action-update-interval", type=int, default=1, help="CPG update interval (env steps); 1 = every env step (match train)")
    p.add_argument("--smoothing-alpha", type=float, default=1, help="EMA smoothing for CPG params (match train)")
    p.add_argument("--max-episode-steps", type=int, default=4100, help="Env episode cap (sim steps); 2050 = 41 env steps × 50 (match train); use larger for long eval)")
    p.add_argument("--n-substeps", type=int, default=50, help="Substeps per env step (K)")

    p.add_argument("--vel-weight", type=float, default=1.0, help="Velocity tracking weight")
    p.add_argument("--pose-weight", type=float, default=0.1, help="Stability penalty weight (match train)")
    p.add_argument("--smooth-weight", type=float, default=1e-3, help="Action smoothness weight")
    p.add_argument("--energy-weight", type=float, default=0.05, help="Energy weight (match train)")
    p.add_argument("--cot-weight", type=float, default=0.0, help="CoT weight (match train)")
    p.add_argument("--x-weight", type=float, default=0.2, help="X deviation penalty weight (match train)")
    p.add_argument("--ik-fail-weight", type=float, default=0.2, help="IK failure penalty weight")

    p.add_argument("--randomize-cpg-params", action="store_true", help="Randomize CPG f/mu at reset (match train if used)")
    p.add_argument("--reset-f-sigma", type=float, default=0.05, help="Std for initial f perturbation when --randomize-cpg-params (match train)")
    p.add_argument("--use-phase-for-contact", action="store_true", help="Use CPG phase for contact/stance (φ≥π), align with sim_test; else sensordata")
    p.add_argument("--fixed-cpg", action="store_true", help="Ignore policy action, always use f4=0.5 Hz and mu4=1.0 (same as sim_test); for verifying env forward motion")
    p.add_argument("--use-energy-tank", action="store_true", help="Use energy tank (match train if model was trained with --use-energy-tank); obs +2 dims")

    p.add_argument("--save-csv", action="store_true", help="Enable per-step CSV debug log")
    p.add_argument("--csv-path", type=str, default="logs/test_csv/rat_debug.csv", help="CSV output path (when --save-csv)")
    p.add_argument(
        "--save-episode-metrics",
        action="store_true",
        help="Save per-episode summary metrics (distance, energy, etc.) to CSV",
    )
    p.add_argument(
        "--episode-metrics-path",
        type=str,
        default="logs/eval_episode_metrics.csv",
        help="CSV path for per-episode metrics (when --save-episode-metrics)",
    )
    p.add_argument("--save-cpg-foot-csv", action="store_true", help="Save CPG phase, amplitude and foot target (Fy,Fz) per sim step to CSV for gait visualization")
    p.add_argument("--cpg-foot-csv-path", type=str, default="logs/eval_cpg_foot.csv", help="Output path for CPG/foot CSV (when --save-cpg-foot-csv)")
    p.add_argument("--print-sim-step-joint-angles", action="store_true", help="Print target/actual joint angles every sim step")
    p.add_argument("--print-sim-step-torque", action="store_true", help="Print motor torque every sim step")
    p.add_argument("--print-data-ctrl", action="store_true", help="Print data.ctrl (送入仿真) every sim step (callback)")
    p.add_argument("--print-fre-mu", action="store_true", help="Print CPG f (Hz) and mu every env step")

    return p.parse_args()


def _maybe_launch_viewer(env) -> Optional[object]:
    try:
        import mujoco.viewer as mj_viewer
        return mj_viewer.launch_passive(env.model, env.data)
    except Exception as e:
        print(f"[Eval] Viewer unavailable: {e}")
        return None


def _mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = set()
    for d in dicts:
        keys |= set(d.keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def _get_actual_joint_positions(env) -> List[Tuple[str, int, float]]:
    """Return list of (joint_name, index, qpos_value) for the 8 controlled joints."""
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    # controlled_joint_names: (hip, knee) per leg -> 8 joints in order [leg0_hip, leg0_knee, ...]
    flat_names: List[str] = []
    for hip_name, knee_name in unwrapped.controlled_joint_names:
        flat_names.append(hip_name)
        flat_names.append(knee_name)
    out: List[Tuple[str, int, float]] = []
    for idx, (hip_id, knee_id) in enumerate(unwrapped.joint_ids):
        for j, jid in enumerate([hip_id, knee_id]):
            global_idx = idx * 2 + j
            if jid >= 0:
                qpos_adr = int(model.jnt_qposadr[jid])
                val = float(data.qpos[qpos_adr])
                out.append((flat_names[global_idx], global_idx, val))
    return out


def _flat_joint_names(env) -> List[str]:
    """Return 8 joint names in order [leg0_hip, leg0_knee, ...]."""
    flat: List[str] = []
    for hip_name, knee_name in env.unwrapped.controlled_joint_names:
        flat.append(hip_name)
        flat.append(knee_name)
    return flat


def _print_joint_targets_and_actual(
    joint_targets: List[Tuple[float, float]],
    actual_list: List[Tuple[str, int, float]],
    env_step: int,
) -> None:
    """Print target joint angles (index + value) and actual joint angles (index + value)."""
    names_by_idx = {idx: name for (name, idx, _) in actual_list}
    print(f"  [env_step={env_step}] 目标关节角度 (target):")
    for leg_i in range(4):
        q1_des, q2_des = joint_targets[leg_i]
        print(f"    index {leg_i*2}: {names_by_idx[leg_i*2]} = {q1_des:.5f} rad")
        print(f"    index {leg_i*2+1}: {names_by_idx[leg_i*2+1]} = {q2_des:.5f} rad")
    print(f"  [env_step={env_step}] 实际关节角度 (actual):")
    for name, idx, val in sorted(actual_list, key=lambda x: x[1]):
        print(f"    index {idx}: {name} = {val:.5f} rad")


def _print_joint_targets_and_actual_sim_step(
    sim_step: int,
    joint_targets: List[Tuple[float, float]],
    actual_qpos_8: List[float],
    flat_names: List[str],
) -> None:
    """Print target vs actual at one sim step (callback from env)."""
    names_by_idx = {i: flat_names[i] for i in range(8)}
    print(f"  [sim_step={sim_step}] 目标关节角度 (target):")
    for leg_i in range(4):
        q1_des, q2_des = joint_targets[leg_i]
        print(f"    index {leg_i*2}: {names_by_idx[leg_i*2]} = {q1_des:.5f} rad")
        print(f"    index {leg_i*2+1}: {names_by_idx[leg_i*2+1]} = {q2_des:.5f} rad")
    print(f"  [sim_step={sim_step}] 实际关节角度 (actual):")
    for idx in range(8):
        print(f"    index {idx}: {flat_names[idx]} = {actual_qpos_8[idx]:.5f} rad")


def _print_data_ctrl(env, sim_step: int, joint_targets: List[Tuple[float, float]], flat_names: List[str]) -> None:
    """Print data.ctrl (送入仿真的控制量) and compare with joint_targets."""
    unwrapped = env.unwrapped
    print(f"  [sim_step={sim_step}] data.ctrl (送入仿真):")
    for leg_i in range(4):
        hip_act, knee_act = unwrapped.actuator_ids[leg_i]
        q1_des, q2_des = joint_targets[leg_i]
        ctrl_hip = float(unwrapped.data.ctrl[hip_act])
        ctrl_knee = float(unwrapped.data.ctrl[knee_act])
        print(
            f"    {flat_names[leg_i*2]:20s} act[{hip_act}] = {ctrl_hip:+.5f} rad  (target = {q1_des:+.5f})"
        )
        print(
            f"    {flat_names[leg_i*2+1]:20s} act[{knee_act}] = {ctrl_knee:+.5f} rad  (target = {q2_des:+.5f})"
        )


def main() -> None:
    args = parse_args()

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
        cot_weight=args.cot_weight,
        x_weight=args.x_weight,
        ik_fail_weight=args.ik_fail_weight,
        randomize_cpg_params=args.randomize_cpg_params,
        reset_f_sigma=args.reset_f_sigma,
        use_phase_for_contact=args.use_phase_for_contact,
        fixed_cpg=args.fixed_cpg,
        use_energy_tank=args.use_energy_tank,
        enable_csv_log=args.save_csv,
        csv_path=args.csv_path,
        auto_timestamp_csv=True,
        print_sim_step_joint_angles=args.print_sim_step_joint_angles,
        print_sim_step_torque=args.print_sim_step_torque,
        render_mode="human" if args.render else None,
    )

    # 评估前打印 obs 中各状态名称及索引（与 train 一致）
    if hasattr(env, "get_obs_layout"):
        print("\n[Eval] Observation layout (obs 中用到的状态及其索引):")
        for name, idx_str in env.get_obs_layout():
            print(f"  obs[{idx_str}]: {name}")
        print(f"  total dim: {env.observation_space.shape[0]}\n")

    if args.save_csv:
        print(f"[Eval substep50] CSV 保存: 已开启, 路径={args.csv_path}")
    else:
        print("[Eval substep50] CSV 保存: 未开启 (使用 --save-csv 开启)")

    viewer = _maybe_launch_viewer(env) if args.render else None
    if viewer is not None:
        env.unwrapped.set_viewer(viewer)

    # 可选：每 sim step 打印 data.ctrl（env 已支持 --print-sim-step-joint-angles / --print-sim-step-torque）
    if args.print_data_ctrl:
        flat_names = _flat_joint_names(env)

        def _sim_step_cb(sim_step: int, joint_targets: List[Tuple[float, float]], actual_qpos_8: List[float]) -> None:
            _print_data_ctrl(env, sim_step, joint_targets, flat_names)

        env.unwrapped.set_sim_step_callback(_sim_step_cb)

    # CPG + 足底轨迹记录（每 sim step 一行），用于 plot_cpg_foot.py
    cpg_foot_rows: List[List] = []
    cpg_foot_csv_path = os.path.abspath(args.cpg_foot_csv_path) if args.save_cpg_foot_csv else None
    if args.save_cpg_foot_csv:
        os.makedirs(os.path.dirname(cpg_foot_csv_path) or ".", exist_ok=True)
        with open(cpg_foot_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["sim_step", "env_step", "episode"]
                + [f"phase_{i}" for i in range(4)]
                + [f"amp_{i}" for i in range(4)]
                + [f"Fy_{i}" for i in range(4)]
                + [f"Fz_{i}" for i in range(4)]
            )
        print(f"[Eval substep50] CPG/足底 CSV: 已开启, 路径={cpg_foot_csv_path}")

        def _cpg_foot_cb(sim_step: int, joint_targets: List[Tuple[float, float]], actual_qpos_8: List[float]) -> None:
            unwrapped = env.unwrapped
            env_step = getattr(unwrapped, "_eval_env_step", 0)
            episode = getattr(unwrapped, "_eval_episode", 0)
            phases = [unwrapped.foot_path.get_leg_phase(j) for j in range(4)]
            amps = [unwrapped.foot_path.get_leg_amp(j) for j in range(4)]
            feet = [unwrapped.foot_path.get_foot_target(j) for j in range(4)]
            row = [sim_step, env_step, episode] + phases + amps + [fy for fy, _ in feet] + [fz for _, fz in feet]
            cpg_foot_rows.append(row)

        _prev_cb = env.unwrapped.sim_step_callback

        def _chained_cb(sim_step: int, joint_targets: List[Tuple[float, float]], actual_qpos_8: List[float]) -> None:
            if _prev_cb is not None:
                _prev_cb(sim_step, joint_targets, actual_qpos_8)
            _cpg_foot_cb(sim_step, joint_targets, actual_qpos_8)

        env.unwrapped.set_sim_step_callback(_chained_cb)

    # 若模型为旧版 43 维 obs，当前 env 为 35 维，则用包装器填充 joint_vel 以兼容
    try:
        model = PPO.load(args.model, env=env)
    except ValueError as e:
        err_msg = str(e)
        if "43" in err_msg and "35" in err_msg and "Observation spaces" in err_msg:
            print("[Eval] 检测到模型为旧版 43 维 obs，当前 env 为 35 维；使用 ObsPadWrapper35to43 填充以兼容。")
            env = ObsPadWrapper35to43(env)
            model = PPO.load(args.model, env=env)
        else:
            raise

    total_rewards = []
    episode_rows: List[List] = []
    metrics_csv_path = os.path.abspath(args.episode_metrics_path) if args.save_episode_metrics else None
    if args.save_episode_metrics and metrics_csv_path is not None:
        os.makedirs(os.path.dirname(metrics_csv_path) or ".", exist_ok=True)
        if not os.path.exists(metrics_csv_path):
            with open(metrics_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "episode",
                        "steps",
                        "reward",
                        "fwd_dist_pos_dy",
                        "fwd_dist_raw_dy",
                        "x_offset_m",
                        "ep_abs_work_J",
                        "ep_abs_power_mean_W",
                        "ep_energy_J",
                        "stance_abs_work_total_J",
                        "swing_abs_work_total_J",
                        "terminated",
                        "truncated",
                    ]
                )
    for ep in range(args.episodes):
        cpg_foot_rows.clear()
        reset_options = {}
        if args.seed is not None:
            reset_options["seed"] = args.seed + ep  # different seed per episode if multiple
        obs, _ = env.reset(**reset_options)
        terminated = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        interval_summaries: List[Dict[str, float]] = []
        cached_action = None  # 做法B: action_update_interval > 1 时，每 N 步才更新
        env.unwrapped._eval_episode = ep

        while not terminated and not truncated and steps < args.max_steps:
            if steps % args.action_update_interval == 0 or cached_action is None:
                cached_action, _ = model.predict(obs, deterministic=True)
            action = cached_action
            env.unwrapped._eval_env_step = steps
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

            if args.print_fre_mu and isinstance(info, dict):
                f4 = info.get("action_f4")
                mu4 = info.get("action_mu4")
                fwd_vel = info.get("fwd_vel")
                if f4 is not None and mu4 is not None:
                    print(f"  [env_step={steps}] fre (Hz): [{float(f4[0]):.4f}, {float(f4[1]):.4f}, {float(f4[2]):.4f}, {float(f4[3]):.4f}]  mu: [{float(mu4[0]):.4f}, {float(mu4[1]):.4f}, {float(mu4[2]):.4f}, {float(mu4[3]):.4f}]")
                if fwd_vel is not None:
                    print(f"    y轴基座速度(前向 fwd_vel)={float(fwd_vel):.6f} m/s")

            if isinstance(info, dict) and "log_interval_means" in info:
                d = info["log_interval_means"]
                if isinstance(d, dict):
                    interval_summaries.append({k: float(v) for k, v in d.items()})

        total_rewards.append(ep_reward)

        if args.save_cpg_foot_csv and cpg_foot_csv_path and cpg_foot_rows:
            with open(cpg_foot_csv_path, "a", newline="") as f:
                csv.writer(f).writerows(cpg_foot_rows)

        ep_interval_mean = _mean_dict(interval_summaries)

        # episode-level metrics from env.info["episode_components"] (only valid at done)
        ep_comp = info.get("episode_components", {}) if isinstance(info, dict) else {}
        if args.save_episode_metrics and metrics_csv_path is not None:
            fwd_pos = ep_comp.get("ep_distance_pos_dy", float("nan"))
            fwd_raw = ep_comp.get("ep_distance_raw", float("nan"))
            x_offset_m = ep_comp.get("x_offset_m", float("nan"))
            ep_abs_work = ep_comp.get("ep_abs_work_J", float("nan"))
            ep_abs_power_mean = ep_comp.get("ep_abs_power_mean_W", float("nan"))
            ep_energy = ep_comp.get("ep_energy_J", float("nan"))
            stance_E = ep_comp.get("stance_abs_work_total_J", float("nan"))
            swing_E = ep_comp.get("swing_abs_work_total_J", float("nan"))
            row = [
                ep + 1,
                steps,
                ep_reward,
                fwd_pos,
                fwd_raw,
                x_offset_m,
                ep_abs_work,
                ep_abs_power_mean,
                ep_energy,
                stance_E,
                swing_E,
                bool(ep_comp.get("terminated", terminated)),
                bool(ep_comp.get("truncated", truncated)),
            ]
            episode_rows.append(row)

        last_fwd = info.get("fwd_vel")
        print(
            f"Episode {ep+1}: reward={ep_reward:.3f}, steps={steps}, "
            f"last_base_vx={info.get('base_vx', None)}"
        )
        print(f"  y轴基座速度(前向 fwd_vel, 本步均值): {last_fwd:.6f}" if last_fwd is not None else "  y轴基座速度(前向 fwd_vel): N/A")

        if ep_interval_mean:
            print(
                f"  [interval-avg] "
                f"mean_reward={ep_interval_mean.get('reward', float('nan')):+.4f}  "
                f"mean_r_vel={ep_interval_mean.get('r_vel', float('nan')):+.4f}  "
                f"mean_r_xvel={ep_interval_mean.get('r_xvel', float('nan')):+.4f}  "
                f"mean_fwd_vel={ep_interval_mean.get('fwd_vel', float('nan')):+.4f}  "
                f"mean_dy={ep_interval_mean.get('dy', float('nan')):+.5f}  "
                f"mean_ik_fail_rate={ep_interval_mean.get('ik_fail_rate', float('nan')):.3f}"
            )

    # 写 episode 级别指标 CSV
    if args.save_episode_metrics and metrics_csv_path is not None and episode_rows:
        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerows(episode_rows)
        print(f"[Eval substep50] Episode metrics CSV: 已保存到 {metrics_csv_path}")

    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    print(f"Average reward over {args.episodes} episodes: {avg_reward:.3f}")

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
