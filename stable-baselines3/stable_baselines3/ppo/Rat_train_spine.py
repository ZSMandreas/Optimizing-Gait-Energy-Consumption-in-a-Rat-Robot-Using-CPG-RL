# -*- coding: utf-8 -*-
"""
训练脚本（方案1）：观测前 72 维与旧模型完全一致，仅在末尾追加 spine_last_action（1维）。
因此：
- VecNormalize：若旧文件是 72 维，载入失败则新建并把前 72 维统计拷贝到新 73 维对象里（最后一维 0/1）。
- 网络迁移：MLP 第一层输入 72→73（前 72 列拷贝，最后一列置零）；动作头 8→9（前 8 行拷贝，第 9 行置零）。
"""

import os
import glob
import argparse
import numpy as np
os.chdir("/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalizeClass
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from ppo import LoggingCallback
from Rat_Env_spine_train import Go2Env


# ========== Callbacks ==========
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.prefix = prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            vec_env = self.training_env
            if isinstance(vec_env, VecNormalizeClass):
                step = self.num_timesteps
                vn_path = os.path.join(self.save_path, f"{self.prefix}_{step}_vecnormalize.pkl")
                vec_env.save(vn_path)
                if self.verbose > 0:
                    print(f"[SaveVecNormalizeCallback] Saved VecNormalize → {vn_path}")
        return True


def latest_checkpoint(root: str, prefix: str):
    zips = sorted(glob.glob(os.path.join(root, f"{prefix}_*_steps.zip")))
    if not zips:
        return None, None
    latest_zip = zips[-1]
    step = latest_zip.split("_")[-2]
    vn = os.path.join(root, f"{prefix}_{step}_vecnormalize.pkl")
    if not os.path.exists(vn):
        vn = None
    return latest_zip, vn


# ========== VecNormalize 迁移 ==========
def migrate_vecnormalize_or_fresh(vec_env, vn_path: str | None):
    loaded_ok = False
    if vn_path is not None and os.path.exists(vn_path):
        try:
            vec_env = VecNormalize.load(vn_path, vec_env)
            print(f"[INFO] Loaded VecNormalize statistics from: {vn_path}")
            loaded_ok = True
        except AssertionError as e:
            print(f"[WARN] VecNormalize shape mismatch: {e}. Will recreate and migrate 72→73.")

    if not loaded_ok:
        vec_env = VecNormalize(vec_env, norm_obs=True)
        print("[INFO] Using fresh VecNormalize (new obs dim = 73).")
        # 尝试把旧文件的前72维统计迁移到新对象
        if vn_path is not None and os.path.exists(vn_path):
            try:
                import cloudpickle as pickle
                with open(vn_path, "rb") as f:
                    old_vn = pickle.load(f)
                old_mean, old_var, old_count = old_vn.obs_rms.mean, old_vn.obs_rms.var, old_vn.obs_rms.count
                new_mean, new_var = vec_env.obs_rms.mean, vec_env.obs_rms.var
                if new_mean.shape[0] == old_mean.shape[0] + 1:
                    new_mean[:old_mean.shape[0]] = old_mean
                    new_var[:old_var.shape[0]] = old_var
                    new_mean[-1] = 0.0
                    new_var[-1] = 1.0
                    vec_env.obs_rms.count = old_count
                    print(f"[INFO] Migrated VecNormalize stats: 72 -> 73 (last dim init 0/1).")
            except Exception as e:
                print(f"[WARN] Could not migrate VecNormalize stats: {e}")
    return vec_env


# ========== 权重迁移（72→73 输入，8→9 动作头） ==========
def transplant_mlp_first_layer(old_model, new_model):
    import torch
    with torch.no_grad():
        old_mlp = old_model.policy.mlp_extractor
        new_mlp = new_model.policy.mlp_extractor

        # policy 分支第一层 Linear
        old_pol_fc1 = old_mlp.policy_net[0]
        new_pol_fc1 = new_mlp.policy_net[0]
        in_old = old_pol_fc1.weight.shape[1]  # 72
        in_new = new_pol_fc1.weight.shape[1]  # 73
        if in_new != in_old + 1:
            raise RuntimeError(f"Unexpected dims policy fc1: old_in={in_old}, new_in={in_new}")
        new_pol_fc1.weight.zero_()
        new_pol_fc1.bias.copy_(old_pol_fc1.bias)
        new_pol_fc1.weight[:, :in_old].copy_(old_pol_fc1.weight)

        # value 分支第一层 Linear
        old_val_fc1 = old_mlp.value_net[0]
        new_val_fc1 = new_mlp.value_net[0]
        in_old_v = old_val_fc1.weight.shape[1]  # 72
        in_new_v = new_val_fc1.weight.shape[1]  # 73
        if in_new_v != in_old_v + 1:
            raise RuntimeError(f"Unexpected dims value fc1: old_in={in_old_v}, new_in={in_new_v}")
        new_val_fc1.weight.zero_()
        new_val_fc1.bias.copy_(old_val_fc1.bias)
        new_val_fc1.weight[:, :in_old_v].copy_(old_val_fc1.weight)

        # 其余层 strict=False 加载
        def _strip_first_linear(sd, prefix):
            for k in [f"{prefix}.0.weight", f"{prefix}.0.bias"]:
                sd.pop(k, None)
            return sd

        old_sd = old_mlp.state_dict()
        old_sd = _strip_first_linear(old_sd, "policy_net")
        old_sd = _strip_first_linear(old_sd, "value_net")
        new_mlp.load_state_dict(old_sd, strict=False)


def transplant_action_head_8_to_9(old_model, new_model):
    import torch
    with torch.no_grad():
        old_w = old_model.policy.action_net.weight
        old_b = old_model.policy.action_net.bias
        new_w = new_model.policy.action_net.weight
        new_b = new_model.policy.action_net.bias
        # SB3: weight [action_dim, latent_pi], bias [action_dim]
        if new_w.shape[0] == old_w.shape[0] + 1 and new_w.shape[1] == old_w.shape[1]:
            new_w.zero_(); new_b.zero_()
            new_w[:old_w.shape[0], :].copy_(old_w)
            new_b[:old_b.shape[0]].copy_(old_b)
        else:
            # 保险处理（若维度转置）
            new_w.zero_(); new_b.zero_()
            new_w[:, :old_w.shape[1]].copy_(old_w)
            new_b[:old_b.shape[0]].copy_(old_b)


def build_new_model(old_model_path: str,
                    new_env: VecNormalize,
                    device: str = "cpu",
                    learning_rate: float = 1e-4,
                    use_sde: bool = True,
                    log_std_init: float = 0.0,
                    squash_output: bool = True):
    # 旧模型（8-dim 动作，72-dim 观测）
    old_model = PPO.load(old_model_path, device=device)

    # 新模型（9-dim 动作，73-dim 观测）
    policy_kwargs = dict(log_std_init=log_std_init, squash_output=squash_output)
    new_model = PPO(
        "MlpPolicy",
        new_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        learning_rate=learning_rate,
        device=device,
        use_sde=use_sde,
        policy_kwargs=policy_kwargs
    )

    # features
    new_model.policy.features_extractor.load_state_dict(
        old_model.policy.features_extractor.state_dict()
    )
    # MLP 第一层手工 72→73，其他层 strict=False
    transplant_mlp_first_layer(old_model, new_model)
    # value 头维度不变
    new_model.policy.value_net.load_state_dict(
        old_model.policy.value_net.state_dict()
    )
    # action 头 8→9
    transplant_action_head_8_to_9(old_model, new_model)

    return new_model


# ========== main ==========
def main(args: argparse.Namespace):
    SAVE_ROOT = args.save_dir
    os.makedirs(SAVE_ROOT, exist_ok=True)
    SAVE_FREQ = args.save_freq
    PREFIX = args.prefix

    # squash_output 需要配合 gSDE
    if args.squash_output and not args.use_sde:
        print("[INFO] squash_output=True requires use_sde=True; enabling gSDE automatically.")
        args.use_sde = True

    # 旧 checkpoint / vecnorm
    model_path = args.model
    vn_path = args.vecnorm
    if model_path is None:
        model_path, vn_auto = latest_checkpoint(args.ckpt_dir, args.prefix)
        if model_path is None:
            model_path = os.path.join(args.ckpt_dir, f"{args.prefix}_final.zip")
            vn_auto = os.path.join(args.ckpt_dir, f"{args.prefix}_final_vecnormalize.pkl")
            print("[WARN] No step checkpoint found; fallback to FINAL model.")
        if vn_path is None:
            vn_path = vn_auto

    # 新 env（观测 73）
    base_env_fn = lambda: Go2Env(render_mode="window",
                                 spine_act_index=args.spine_act_index,
                                 spine_scale=args.spine_scale)
    vec_env = make_vec_env(base_env_fn, n_envs=1, seed=args.seed)

    # VecNormalize：尝试加载，不符则新建并迁移 72→73
    vec_env = migrate_vecnormalize_or_fresh(vec_env, vn_path)
    vec_env.training = False
    vec_env.norm_reward = False

    # 新 PPO + 权重迁移
    model = build_new_model(
        old_model_path=model_path,
        new_env=vec_env,
        device=args.device,
        learning_rate=args.learning_rate,
        use_sde=args.use_sde,
        log_std_init=args.log_std_init,
        squash_output=args.squash_output
    )

    # 回调
    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=SAVE_ROOT,
        name_prefix=PREFIX,
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    save_vn_cb = SaveVecNormalizeCallback(
        save_freq=SAVE_FREQ,
        save_path=SAVE_ROOT,
        prefix=PREFIX,
        verbose=1
    )
    logging_cb = LoggingCallback(verbose=1)
    callbacks = CallbackList([checkpoint_cb, save_vn_cb, logging_cb])

    # 训练
    vec_env.training = True
    vec_env.norm_reward = False
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # 收尾保存
    final_model = os.path.join(SAVE_ROOT, f"{PREFIX}_final.zip")
    model.save(final_model)
    final_vn = os.path.join(SAVE_ROOT, f"{PREFIX}_final_vecnormalize.pkl")
    vec_env.save(final_vn)
    print(f"[DONE] Saved final model: {final_model}")
    print(f"[DONE] Saved final VecNormalize: {final_vn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Old model checkpoint dir")
    parser.add_argument("--prefix", type=str, default="rat_mapping", help="Old checkpoint file prefix")
    parser.add_argument("--model", type=str, default=None, help="Path to a specific old model .zip (optional)")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to a specific VecNormalize .pkl (optional)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_spine", help="New checkpoints dir")
    parser.add_argument("--save_freq", type=int, default=25_000, help="Checkpoint save freq (steps)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_sde", action="store_true", help="Enable generalized State-Dependent Exploration")
    parser.add_argument("--log_std_init", type=float, default=-1.0, help="Initial log standard deviation")
    parser.add_argument("--squash_output", action="store_true", help="Use tanh squashing (requires --use_sde)")
    parser.add_argument("--spine_act_index", type=int, default=8, help="Spine actuator index in Mujoco ctrl")
    parser.add_argument("--spine_scale", type=float, default=0.4, help="Scale raw spine action [-1,1] to radians")
    args = parser.parse_args()
    # 若未显式给 squash_output，则保持 True 以匹配你原训练习惯
    if not args.squash_output:
        args.squash_output = True
    main(args)
