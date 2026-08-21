import numpy as np
import gym
from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
# import glfw
import numpy as np
from scipy.signal import butter, lfilter
import pandas as pd
import time
import matplotlib.pyplot as plt
# from GO2state import State
from Remostate import State

    
    
class Go2Env(gym.Env):
    def __init__(self, render_mode=None,xml_file_path="/home/simiao/semester arbeit/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",max_steps=1024, goal_vector=None):
        super(Go2Env, self).__init__()
        
        # 加载 MuJoCo 模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        # 渲染器初始化
        # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer = None
        self.previous_reward = 0
        self.seed_value = None
        self.n = 5
        self.state = State(self.model, self.data, self.n)
        self.goal_vector = goal_vector if goal_vector is not None else np.array([1.0, 0.0, 0.5])  # 默认向前 0.5m/s
        
        # 设置摄像头跟踪基座
        base_body_name = "base"  # 假设机器人基座的名称是 "base"
        try:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")
        
        # 环境的关节和电机控制参数设置
        self.n_joints = 8
        # self.state_dim = self.n_joints * 2  
        self.action_dim = self.n_joints     # 每个关节的控制力矩
        
        #窗口
        # self.obs_window_size = 5
        # self.action_window_size = 2
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []

        # # 定义关节和电机控制范围
        # self.joint_ranges = [
        #     (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),  # FL
        #     (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),  # FR
        #     (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776),  # RL
        #     (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776)   # RR
        # ]
        self.motor_ranges = [
            (-0.157, 0.157), (-0.157, 0.157),  # FL
            (-0.157, 0.157), (-0.157, 0.157),  # FR
            (-0.157, 0.157), (-0.157, 0.157),  # RL
            (-0.157, 0.157), (-0.157, 0.157),  # RR
        ]
        # self.motor_ranges = [
        #     (-0.157, 0.157), (-0.4, 0.4),  # FL
        #     (-0.157, 0.157), (-0.4, 0.4),  # FR
        #     (-0.157, 0.157), (-0.4, 0.4),  # RL
        #     (-0.157, 0.157), (-0.4, 0.4),  # RR
        # ]
        
        contact_force_low = np.full(4, -10)  # feet condition
        contact_force_high = np.full(4, 10)
        
        
        low_obs = np.concatenate((
            np.full(self.n_joints, -np.pi), #关节角度下界 
            # np.array([m[0] for m in self.motor_ranges]),
            np.full(4, -np.pi), # other angles
            contact_force_low,
            np.full(3, -np.inf), # base position
            np.full(4, -np.pi),#euler
            np.full(3, -np.inf),  # 基座线速度
            np.full(6, -np.inf),  # 基座加速度和角速度      
        ))

        high_obs = np.concatenate((
            np.full(self.n_joints, np.pi), #关节角度下界
            # np.array([m[1] for m in self.motor_ranges]),
            np.full(4, np.pi), # other angles
            contact_force_high,
            np.full(3, np.inf), # base position
            np.full(4, np.pi),#euler
            np.full(3, np.inf),  # 基座线速度
            np.full(6, np.inf),  # 基座加速度和角速度  
        ))
            
        # 单步动作上下界
        # low_action = np.array([m[0] for m in self.motor_ranges])
        # high_action = np.array([m[1] for m in self.motor_ranges])
        low_action = np.concatenate((
            # np.full(self.n_joints, -1.0),#forcerange [-0.157 0.157]
            np.array([m[0] for m in self.motor_ranges]),
            # np.full(1, -0.6),#tail
            np.full(1, -0.157)#head spine
        ))
        high_action = np.concatenate((
            # np.full(self.n_joints, 1.0),#forcerange [-0.157 0.157]
            np.array([m[1] for m in self.motor_ranges]),
            # np.full(1, 0.6),#tail
            np.full(1, 0.157)#head spine
        ))

        goal_low = np.array([-1.0, -1.0, -3.0])   # 假设方向是单位向量，速度 >= 0
        goal_high = np.array([1.0, 1.0, 3.0])    # 可根据实际目标速度范围设置
        
        # 拼接扩展后的上下界
        final_low = np.concatenate((low_obs,low_action,low_obs,goal_low))
        final_high = np.concatenate((high_obs,high_action,high_obs,goal_high))

        # 更新观测空间
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        
        # 动作空间：归一化为 [-1, 1]
        self.action_space = spaces.Box(low=low_action, high=high_action, shape=(9,), dtype=np.float32)

        
    
    
    def reset(self, **kwargs):
        self.current_step = 0
        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)

        # 每次 reset 时重新采样一个 goal_vector
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(-1.2, 1.2)
        self.goal_vector = np.array([np.cos(angle), np.sin(angle), speed])

        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        st = self.state.get_observation()
        mujoco.mj_forward(self.model, self.data)
        st_ = self.state.get_observation()
        state = np.concatenate([st, self.data.ctrl, st_, self.goal_vector])
        return state, {}

    def step(self, action):
        for _ in range(self.n):
            if len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    mujoco.mj_step(self.model, self.data)
                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)

            mujoco.mj_step(self.model, self.data)
            obs = self.state.get_observation()
            self.state.update_next_observations(obs)

        state = self.state.get_state(action)
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()

        # 将 goal 向量加入状态拼接
        state = np.concatenate([state, self.goal_vector])

        reward, reward_info = self.calculate_rewards(state)
        done = False
        if self.has_fallen():
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        return state, reward, done, truncated, reward_info

    def calculate_rewards(self, state):
        """
        计算奖励：结合方向对齐、速度匹配、稳定性等
        """
        def sigmoid_scaled(x, scale=1.0):
            return 2 / (1 + np.exp(-scale * x)) - 1  # ∈ (-1, 1)
        
        base_vel = state[64:67]  # 线速度 vx, vy, vz
        vx, vy = base_vel[0], base_vel[1]

        # --- (1) 方向奖励 ---
        goal_dir = self.goal_vector[:2] / (np.linalg.norm(self.goal_vector[:2]) + 1e-6)
        current_dir = np.array([vx, vy])
        r_dir = np.dot(current_dir, goal_dir)  # ∈ (-inf, inf)
        r_dir_norm = sigmoid_scaled(r_dir, scale=1.0)  # scale 可调节曲线陡峭程度

        # --- (2) 速度匹配奖励 ---
        current_speed = np.linalg.norm(current_dir)
        target_speed = self.goal_vector[2]
        speed_diff = current_speed - target_speed
        r_speed = -abs(speed_diff)  # 原始形式
        r_speed_norm = sigmoid_scaled(-abs(speed_diff), scale=5.0)

        # # --- (3) 姿态稳定性（四元数 → 欧拉角） ---
        # quat = state[60:64]  # [w, x, y, z]
        # euler = R.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler('xyz')
        # roll, pitch, _ = euler  # 我们暂时不需要 yaw
        # base_height = state[59]
        # r_stable_h = 1 - min(abs(base_height - 0.2) / 0.2, 1.0)
        # r_stable_p = 1 - min(abs(pitch) / 0.3, 1.0)
        # r_stable_r = 1 - min(abs(roll) / 0.3, 1.0)
        # r_stable_norm = (r_stable_h + r_stable_p + r_stable_r) / 3

        reward = 1.0 * r_dir_norm + 0.5 * r_speed_norm 

        reward_info = {
            "r_dir": r_dir_norm,
            "r_speed": r_speed_norm,
            # "r_stable": r_stable_norm,
            "total_reward": reward
        }

        return reward, reward_info

    def has_fallen(self):
        """
        改进版机器狗摔倒检测，放宽条件。

        Parameters:
            data: MuJoCo 的 MjData 对象

        Returns:
            bool: True 表示摔倒,False 表示未摔倒
        """
        # 获取基座位置和姿态
        base_position = self.data.sensordata[16:19]  # 基座位置 (x, y, z)
        # quaternion = self.data.qpos[3:7]    # 基座四元数 [w, x, y, z]

        # 判断质心高度是否过低
        com_z = base_position[2]
        min_height = 0.035  # 动态调整阈值
        if com_z < min_height:  # 允许稍微低的质心高度
            return True
        return False
    
    def render(self, mode="human"):
        if mode == "human":
            # if self.viewer is None:
            #     # 初始化渲染器
            #     self.viewer = mujoco.MjRenderContextOffscreen(self.model, self.data)
                


             # 初始化摄像头设置
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                self.viewer.cam.trackbodyid = self.base_body_id  # 设置摄像头跟踪目标
                self.viewer.cam.distance = 2.0   # 摄像机与目标的距离
                self.viewer.cam.elevation = -10  # 摄像机俯仰角
                self.viewer.cam.azimuth = 90     # 摄像机水平角度
                self.viewer.cam.lookat[:] = self.data.qpos[:3]  # 初始目标设置为基座位置
        
            self.viewer.render()



    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None