# from GO2ENV import Go2Env  # 导入自定义环境
# from MouseEnv import Go2Env
from Rat_Env_cpg import Go2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalizeClass

from ppo import LoggingCallback
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from plot_state import plot_measured
from plot_state import VarianceTracker
from Newpolicy import SineParamActorCriticPolicy
from Remostate import State
import mujoco
import os
os.chdir("/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")
import glob
import torch  # <<< 新增：用于逐维设置 log_std
from stable_baselines3.common.utils import get_linear_fn

# =================== 追加：保存 VecNormalize 参数的回调 ===================
class SaveVecNormalizeCallback(BaseCallback):
    """
    每隔 save_freq steps 保存一次 VecNormalize 的归一化参数（.pkl），
    用于测试阶段与模型 checkpoint 成对加载。
    """
    def __init__(self, save_freq: int, save_path: str, prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.prefix = prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            vec_env = self.training_env  # SB3 会把 VecNormalize 暴露为 training_env
            if isinstance(vec_env, VecNormalizeClass):
                step = self.num_timesteps
                vn_path = os.path.join(self.save_path, f"{self.prefix}_{step}_vecnormalize.pkl")
                vec_env.save(vn_path)
                if self.verbose > 0:
                    print(f"[SaveVecNormalizeCallback] Saved VecNormalize → {vn_path}")
        return True


def latest_checkpoint(root: str, prefix: str):
    """找到最新的 checkpoint（zip），并推断对应的 vecnormalize（pkl）。"""
    zips = sorted(glob.glob(os.path.join(root, f"{prefix}_*_steps.zip")))
    if not zips:
        return None, None
    latest_zip = zips[-1]
    # 文件名形如：{prefix}_{step}_steps.zip
    step = latest_zip.split("_")[-2]
    vn = os.path.join(root, f"{prefix}_{step}_vecnormalize.pkl")
    if not os.path.exists(vn):
        vn = None
    return latest_zip, vn


if __name__ == '__main__':
    # =================== 训练配置 ===================
    SAVE_ROOT = "./checkpoints"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    SAVE_FREQ = 5_000           # <<< 每 50k step 保存
    PREFIX    = "rat_mapping"     # 文件名前缀

    # 包装环境为 VecEnv
    env = make_vec_env(lambda: Go2Env(render_mode="window"), n_envs=1, seed=123)

    # 取出第一个环境（Go2Env 实例），以便外部拿到 state 等对象（按你原逻辑保留）
    raw_env = env.envs[0]
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    state = raw_env.state

    # 观测归一化（如需对 reward 也归一可设 norm_reward=True）
    env = VecNormalize(env, norm_obs=True)

    # 启用 squash（+ 启用 SDE）
    policy_kwargs = dict(log_std_init=0, squash_output=False)
    print(f"squash_output: {policy_kwargs['squash_output']}")

    # ========= 训练阶段：增加 Checkpoint + VecNormalize 保存回调 =========
    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=SAVE_ROOT,
        name_prefix=PREFIX,          # 保存为 ./checkpoints/rat_mapping_{step}_steps.zip
        save_replay_buffer=False,
        save_vecnormalize=False      # 这里不保存，由我们自定义回调单独保存 .pkl
    )
    save_vn_cb = SaveVecNormalizeCallback(
        save_freq=SAVE_FREQ,
        save_path=SAVE_ROOT,
        prefix=PREFIX,
        verbose=1
    )
    logging_cb = LoggingCallback(verbose=1)
    callbacks = CallbackList([checkpoint_cb, save_vn_cb, logging_cb])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        learning_rate=get_linear_fn(3e-4, 3e-5,1.0), #线性学习率衰减
        device='cpu',
        use_sde=True,
        n_steps= 64,
        # target_kl=0.2,
        policy_kwargs=policy_kwargs
    )

    # ========= 逐维设置 log_std：前8维=0，spine(第9维)= -1 =========
    with torch.no_grad():
        ls = model.policy.log_std  # gSDE: (features_dim, action_dim); 非gSDE: (action_dim,)
        if ls.ndim == 1:
            # 非 gSDE
            assert ls.numel() >= 8, f"动作维度 {ls.numel()} < 9？"
            ls[:8] = -1.0          # 前8维
            # ls[8]  = -1.0         # 第9维（spine）
        elif ls.ndim == 2:
            # gSDE：按“列=动作维”设置
            feat_dim, act_dim = ls.shape
            assert act_dim >= 8, f"动作维度 {act_dim} < 9？"
            ls[:, :8] = -1.0       # 前8列 → 0
            # ls[:,  8] = -1.0      # 第9列（spine） → -1
        else:
            raise RuntimeError(f"Unexpected log_std shape: {ls.shape}")
        # 可选：给逐维/全局下界，避免后期熵被压没
        # model.policy.log_std.data.clamp_(min=-1.5)

    # 打印核对：对特征维求均值后应为 [0,...,0,-1]
    ls_now = model.policy.log_std.data
    per_dim = ls_now if ls_now.ndim == 1 else ls_now.mean(dim=0)
    print("[Per-dim log_std (mean over features)]:", per_dim.cpu().numpy())

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # 训练结束后，保存“最终”一份（方便直接用）
    final_model = os.path.join(SAVE_ROOT, f"{PREFIX}_final.zip")
    model.save(final_model)
    final_vn = os.path.join(SAVE_ROOT, f"{PREFIX}_final_vecnormalize.pkl")
    env.save(final_vn)
    print(f"[DONE] Saved final model: {final_model}")
    print(f"[DONE] Saved final VecNormalize: {final_vn}")
