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
from scipy.spatial.transform import Rotation

    
    
class Go2Env(gym.Env):
    def __init__(self, render_mode=None,xml_file_path="/home/simiao/semester arbeit/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",max_steps=1024):
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
        
        # ctrlrange +-90degree experience
        # self.motor_ranges = [
        #     (-0.785, 0.785), (-0.785, 0.785),
        #     (-0.785, 0.785), (-0.785, 0.785),
        #     (-0.785, 0.785), (-0.785, 0.785),
        #     (-0.785, 0.785), (-0.785, 0.785),
        # ]
        # self.motor_ranges = [
        #     (-0.157, 0.157), (-0.157, 0.157),  # FL
        #     (-0.157, 0.157), (-0.157, 0.157),  # FR
        #     (-0.157, 0.157), (-0.157, 0.157),  # RL
        #     (-0.157, 0.157), (-0.157, 0.157),  # RR
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
            # np.full(4,0)
            # np.full(1, -0.6),#tail
            # np.full(1, -0.157)#head spine
        ))
        high_action = np.concatenate((
            # np.full(self.n_joints, 1.0),#forcerange [-0.157 0.157]
            np.array([m[1] for m in self.motor_ranges]),
            # np.full(4,0)
            # np.full(1, 0.6),#tail
            # np.full(1, 0.157)#head spine
        ))

        # 拼接扩展后的上下界
        final_low = np.concatenate((low_obs,low_action,low_obs))
        final_high = np.concatenate((high_obs,high_action,high_obs))

        # 更新观测空间
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        
        # 动作空间：归一化为 [-1, 1]
        self.action_space = spaces.Box(low=low_action, high=high_action, shape=(8,), dtype=np.float32)

        
    
    
    def reset(self, **kwargs):
        self.current_step = 0
        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)  # 设置种子
        # self.state.initializing()
        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        st = self.state.get_observation()
        print(f"observation{self.data.qpos}")
        print(self.data.qvel)
        # at = self.data.ctrl
        # print(at)
        
        
        mujoco.mj_forward(self.model, self.data)
        st_ = self.state.get_observation()
        # st_diff = st_ - st
        state = np.concatenate([st, self.data.ctrl,st_])
        # print(state)
        return state, {}




    def step(self, action):
        """
        执行一步动作，返回新状态 S、奖励、完成标志等。
        """
        # self.action_delay = []
        self.data.ctrl[:self.n_joints] = action
        # print(f"action{action}")
        torques = self.data.qfrc_actuator
        # print(f"torques{torques}")
    
        # self.step_times = []
       

        # elapsed_time = 0
        # start_time = time.time()
        # for _ in range(5):
        #        if  len(self.state.obs_buffer) == 0:
        #            for _ in range(5):
        #                step_start_time = time.time()  # 每次开始模拟的实际时间
        #                mujoco.mj_step(self.model, self.data)  # 执行一步模拟
        #                for _ in range(5):
        #                     obs_prev = self.state.get_observation()
        #                     self.state.update_observations(obs_prev)
        #                # 控制单步模拟时间
        #                step_end_time = time.time()
        #                elapsed_time = step_end_time-step_start_time
        #                sleep_time = 0.002 - elapsed_time  # 计算补偿的睡眠时间
        #                if sleep_time > 0:
        #                    time.sleep(sleep_time)  # 使模拟的每一步耗时约为 0.002 秒
                     
    
               
    
        #        # obs_buffer = self.state.get_obs_buffer()
        #     #    print(action)       
        #        step_start_time = time.time()  # 每次开始模拟的实际时间
        #        mujoco.mj_step(self.model, self.data)  # 执行一步模拟
        #        for _ in range(5):
        #            obs = self.state.get_observation()
        #            self.state.update_next_observations(obs)
        #        # 控制单步模拟时间
        #        step_end_time = time.time()
        #        elapsed_time = step_end_time-step_start_time
        #        sleep_time = 0.002 - elapsed_time  # 计算补偿的睡眠时间
        #        if sleep_time > 0:
        #            time.sleep(sleep_time)  # 使模拟的每一步耗时约为 0.002 秒
        #     #    print(self.state.next_obs_buffer[:5]) 
        #     #    obs = self.state.get_observation()  # 获取观测值
        #     #    self.state.update_next_observations(obs)
        #        # obs_buffer = self.state.get_next_obs_buffer()
        #        # print(obs_buffer)
        # # print(f"obsbuffer{self.state.obs_buffer[0]}")    
        # # print(f"nextobsbuffer{self.state.next_obs_buffer[0]}")    
        # state = self.state.get_state(action)
        # self.state.obs_buffer = self.state.next_obs_buffer.copy()
        
        for _ in range(self.n):
            if  len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    step_start_time = time.time()  # 每次开始模拟的实际时间
                    mujoco.mj_step(self.model, self.data)  # 执行一步模拟
                    # # 控制单步模拟时间
                    # step_end_time = time.time()
                    # elapsed_time = step_end_time-step_start_time
                    # print(elapsed_time)
                    # sleep_time = 0.002 - elapsed_time  # 计算补偿的睡眠时间
                    # if sleep_time > 0:
                    #     time.sleep(sleep_time)  # 使模拟的每一步耗时约为 0.002 秒
                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)

            

            # obs_buffer = self.state.get_obs_buffer()
            # print(obs_buffer)       
            step_start_time = time.time()  # 每次开始模拟的实际时间
            mujoco.mj_step(self.model, self.data)  # 执行一步模拟
            # # 控制单步模拟时间
            # step_end_time = time.time()
            # elapsed_time = step_end_time-step_start_time
            # sleep_time = 0.002 - elapsed_time  # 计算补偿的睡眠时间
            # if sleep_time > 0:
            #     time.sleep(sleep_time)  # 使模拟的每一步耗时约为 0.002 秒
            obs = self.state.get_observation()  # 获取观测值
            self.state.update_next_observations(obs)
            # obs_buffer = self.state.get_next_obs_buffer()
            # print(obs_buffer)
        # print(f"obsbuffer{self.state.obs_buffer[0]}")    
        # print(f"nextobsbuffer{self.state.next_obs_buffer[0]}")    
        state = self.state.get_state(action)
        self.state.foot_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()
        

        reward, reward_info = self.calculate_rewards(state,action)
        # 判断是否完成
        done = False
        if self.has_fallen():  # 检查是否摔倒
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1  # 更新步数
        # self.current_total_step +=1

        # 计算奖励
        # progress = self.current_total_step / self.total_step
        
        # target_torques = self.data.ctrl[:self.n_joints]  # 输入的目标转矩
        # actual_torques = self.data.actuator_force[:self.n_joints]
        # print(f"target_torques{target_torques}")
        # print(f"actual_torques{actual_torques}")
        # 更新摄像头位置，使其跟随机械狗基座
        if self.render_mode == "window" and self.viewer is not None:
            # 获取基座位置
            base_position = self.data.sensordata[16:19] # 基座位置 (w, x, y, z)

            # 更新摄像机的目标位置
            self.viewer.cam.lookat[:] = base_position
        # print(state)


        return state, reward, done, truncated, reward_info
     

    
    def _fit_state(self, obs_window):
        """
        使用统计学方法（例如均值）拟合三步观测值为单一状态。
        """
        return np.mean(obs_window, axis=0)  # 对每个观测维度取均值
    
    def _generate_S(self, st, at, st_next):
        """
        拼接 st、at 和 st+1 生成最终状态 S。
        """
        return np.concatenate([st, at, st_next]).astype(np.float32)
    
    def _get_foot_contact_states(self):
        """
        获取四足是否与地面接触的状态。
        返回一个布尔值数组，表示每只脚是否接触地面。
        """
        contact_states = []
        foot_geom_names = ['FL', 'FR', 'RL', 'RR']

        for geom_name in foot_geom_names:
            try:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            except Exception as e:
                raise ValueError(f"Could not find geom '{geom_name}': {e}")

            is_contact = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    is_contact = True
                    break

            contact_states.append(is_contact)

        return np.array(contact_states, dtype=np.float32)  # 转为浮点数组以符合状态空间要求


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
        min_height = 0.03  # 动态调整阈值
        if com_z < min_height:  # 允许稍微低的质心高度
            return True
        return False


    
    def calculate_rewards(self,state,action):
        """
        计算机械狗的奖励，包括稳定性、前进、平滑性、方向和探索奖励，支持平滑动态权重。
        """
        # 获取状态
        # base_position = self.state.get_body_position()
        # base_orientation = self.data.qpos[3:7]
        # joint_positions = self.data.qpos[7:]
        # joint_velocities = self.data.qvel[6:]
        current_velocity = state[64] 
        # print(current_velocity)
        # target_x = 0
        # offset = np.abs(base_position[0] - target_x)
        # offset_reward = 2 * (1 / (1 + np.exp(-3 * offset))) - 1

        # (1) 前进奖励
        forward_velocity = - 5 * current_velocity
        # 用 sigmoid 归一化并加入偏移量
        forward_velocity_reward = 2 * (1 / (1 + np.exp(-forward_velocity))) - 1.02  
        if forward_velocity_reward < 0:
            forward_velocity_reward = forward_velocity_reward/1.02
        else:
            forward_velocity_reward = forward_velocity_reward/0.98
            
            
        # === (2) 大动作惩罚 ===
        ctrl_range = 0.785  # ±45度
        lambda_penalty = 0.01  # 可调参数
        action_normalized = action / ctrl_range
        # action_penalty = -lambda_penalty * np.mean(np.square(action_normalized))
        
        # （3）姿态惩罚 pitch
        quat = self.state.get_body_rotation_euler()
        quat = [quat[1], quat[2], quat[3], quat[0]]
        euler = Rotation.from_quat(quat).as_euler('xyz') 
        # print(euler)
        
        pitch = euler[0]
        # stability_penalty = -0.05 * (pitch**2)
        reward = (
            # w2 * stability_reward +
                forward_velocity_reward 
                # action_penalty +
                # stability_penalty
                # offset_reward
                # trot_phase_reward
        #     # w3 * smoothness_penalty +
            # forward_progress_reward +
        #     w5 * direction_reward +
        #     exploration_bonus
        )
        # reward = forward_velocity_reward
        
        reward_info = {
        #  "stability_reward": stability_reward,
         "forward_velocity_reward": forward_velocity_reward,
        #  "action_penalty": action_penalty,
        #  "stability_penalty":stability_penalty
        #  "trot_phase_reward": trot_phase_reward
        #  "smoothness_penalty": smoothness_penalty,
        #  "forward_progress_reward": forward_progress_reward,
        #  "direction_reward": direction_reward,
        #  "exploration_bonus": exploration_bonus,
         }

        return reward,reward_info


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
                # self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True


        
            self.viewer.render()



    def close(self):
        self.state
        self.state.save_foot_trajectory()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
         
        
        

