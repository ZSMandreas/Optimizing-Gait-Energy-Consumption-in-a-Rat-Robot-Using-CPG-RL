# from GO2ENV import Go2Env  # 导入自定义环境
# from MouseEnv import Go2Env
from RatEnv import Go2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
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










if __name__ == '__main__':
    # # 包装环境为 VecEnv 格式
    env = make_vec_env(lambda: Go2Env(render_mode="window"), n_envs=1,seed=123)
    # 取出第一个环境（Go2Env 实例）
    raw_env = env.envs[0]  # type: Go2Env
    # 尝试递归解包直到拿到 Go2Env
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    # 直接访问 Go2Env 中的 state 实例
    state = raw_env.state
    # env = VecNormalize(env, norm_obs=True)
    # # 启用 squash（+可选启用 SDE）
    # policy_kwargs = dict(log_std_init=-3,squash_output=True)
    # # 训练阶段
    # model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./ppo_tensorboard/",learning_rate= 1e-4,device='cpu',use_sde=True,policy_kwargs=policy_kwargs)
    # # log_dir = "./logs"

 
    # #print(f"squash_output: {model.policy.squash_output}")
    # callback = LoggingCallback(verbose=1)                                                                                                                                                         
    # # print("Observation space:", env.observation_space)
    # #print("Action space:", env.action_space)   
    # # print("是否启用了 squash_output:", model.policy.squash_output)
                 
    # model.learn(total_timesteps=3000000,callback=callback)
    # model.save("5_sleep_rat_mapping_1M") 
    # env.save("5_sleep_rat_mapping_1M.pkl")     




    # 测试阶段
    env = VecNormalize.load("5_sleep_rat_normal_1M.pkl", env)                                                                                                            
    model = PPO.load("5_sleep_rat_normal_1M")

    obs = env.reset()
    i = 0
    action_idx = 0
    pos_ms = []
    vel_ms = []
    states_t_list = []
    actions_t_list = []
    states_tp1_list = []
    base_positions = []
    foot_conditions = []
    # ✅ 初始化四个 site 的轨迹记录器
    site_names = ["ankle_fl", "ankle_fr", "ankle_rl", "ankle_rr"]
    site_ids = {
        name: mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    }
    site_trajs = {name: [] for name in site_names}
    while i in range(1000):
        action,_ = model.predict(obs, deterministic=True)
        # print(action)
        actions_t_list.append(action.flatten()) # a_t
        next_obs, reward, done,  info = env.step(action)
        flatten_obs = next_obs.flatten()
        buffer = state.get_next_obs_buffer()
        states_t_list.append(obs.flatten())     # s_t
        states_tp1_list.append(next_obs.flatten())  # s_{t+1}
        obs = next_obs
    
        vel_ms.append(flatten_obs[64:])
        # 记录 site 位置信息
        for name in site_names:
            pos = state.data.site_xpos[site_ids[name]].copy()
            site_trajs[name].append(pos)    
           
       
        env.render()
        # video_recorder.capture_frame()
        i += 1
        # action_idx = (action_idx + 1) % num_actions
        if done:
            obs = env.reset()  

    env.close()





    