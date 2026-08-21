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
from Controller import MouseController
from CPGcontroller import OscillatorLeg
from LegModel.forPath import LegPath
from LegModel.legs import LegModel
    
    
class Go2Env(gym.Env):
    def __init__(self, render_mode=None,xml_file_path="/home/simiao/semester arbeit/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",max_steps=1024):
        super(Go2Env, self).__init__()
        
        # 加载 MuJoCo 模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None
        self.previous_reward = 0
        self.seed_value = None
        self.n = 5
        self.state = State(self.model, self.data, self.n)
        self.t = 0.0
        self.dt = 0.01
        self.controller = MouseController(
            fre = 0.5,
            time_step=self.dt
        )
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]

        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)
        
        self.legs = [
            self.fl_left,
            self.fl_right,
            self.hl_left,
            self.hl_right
        ]
        
        
        
        # 设置摄像头跟踪基座
        base_body_name = "base"  # 假设机器人基座的名称是 "base"
        try:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")
        
        # 环境的关节和电机控制参数设置
        self.n_joints = 8
        self.n_legs = 4
        # self.state_dim = self.n_joints * 2  
        self.action_dim = self.n_legs*3     # 每个振荡器参数
        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = []
        self.last_action = np.zeros(self.action_dim)
        
        #窗口
        # self.obs_window_size = 5
        # self.action_window_size = 2
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = [] 
        self.base_velocity = []

        low_action = np.array([1.0, 0.5, -1.5]*4)
        high_action =np.array([2.0, 4.0, 1.5]*4)
        
        contact_force_low = np.full(4, -10)  # feet condition
        contact_force_high = np.full(4, 10)
        
        cpg_low = np.array([0.5, -5.0, 0.0, 0.0, -np.pi, -2*np.pi]*self.n_legs)
        cpg_high = np.array([2.5, 5.0, 2*np.pi, 5.0, np.pi, 2*np.pi]*self.n_legs)
        
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
            
        # low_action = np.concatenate((
        #     # np.full(self.n_joints, -1.0),#forcerange [-0.157 0.157]
        #     np.array([m[0] for m in self.motor_ranges]),
        #     np.full(4,0)
        #     # np.full(1, -0.6),#tail
        #     # np.full(1, -1.57)#head spine
        # ))
        # high_action = np.concatenate((
        #     # np.full(self.n_joints, 1.0),#forcerange [-0.157 0.157]
        #     np.array([m[1] for m in self.motor_ranges]),
        #     np.full(4,0)
        #     # np.full(1, 0.6),#tail
        #     # np.full(1, 1.57)#head spine
        # ))
        # n = self.n_joints  # 例如 8

        # 分别设置每个参数的范围
        # self.A = 3.0
        # omega_low, omega_high = 0.0, 2 * np.pi
        # phi_low, phi_high = 0.0, 2 * np.pi
        # offset_low, offset_high = -0.5, 0.5

        
        # low = np.concatenate([
        #     # np.full(n, A_low),
        #     np.full(n, omega_low),
        #     np.full(n, phi_low),
        #     # np.full(n, offset_low),
        #     # np.full(1, -1.57),#spine
        # ])
        
        # high = np.concatenate([
        #     # np.full(n, A_high),
        #     np.full(n, omega_high),
        #     np.full(n, phi_high),
        #     # np.full(n, offset_high),
        #     # np.full(1, 1.57),#spine
        # ])


        # 拼接扩展后的上下界
        final_low = np.concatenate((low_obs,low_action,low_obs,cpg_low))
        # print(len(final_low))
        final_high = np.concatenate((high_obs,high_action,high_obs,cpg_high))

        # 更新观测空间
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)


        # 动作空间：归一化为 [-1, 1]
        # self.action_space = spaces.Box(low=low_action, high=high_action, shape=(8,), dtype=np.float32)
        # self.action_space = spaces.Box(low=low, high=high, shape=(33,), dtype=np.float32)#spine
        # self.action_space = spaces.Box(
        #    low=np.array([-0.1, -0.1, 0.02, 0.01] * 4),   # 4条腿 × 每腿4维
        #    high=np.array([0.1, 0.1, 0.12, 0.08] * 4),
        #    shape=(16,),
        #    dtype=np.float32
        # )
        self.action_space = spaces.Box(
            low_action,    # [μ, ω, ψ]
            high_action,
            shape=(12,),
            dtype=np.float32
        )
        
    
    
    def reset(self, **kwargs):
        self.current_step = 0
        self.action_cached = None
        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)  # 设置种子
        # self.state.initializing()
        #ctrlData = 0 
        cpg_state = []
        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = [cpg.get_state() for cpg in self.cpgs]
        for i, cpg in enumerate(self.cpgs):
            init_joint = cpg.get_joint_position(self.legs[i])
            cpg_state.append(init_joint)
        # print(cpg_state)    
        
        cpg_vector = np.concatenate(self.cpgstate)
        self.last_action = np.zeros(self.action_dim)
        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for i in range(1000):
            self.data.ctrl = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0,0,0,0]
            mujoco.mj_step(self.model, self.data)
        # print(self.data.qpos)
        # print(len(self.data.qpos))
        # print(self.data.qvel)
        st = self.state.get_observation()
        # actual_torques = self.data.actuator_force[:self.n_joints]
        # print(f"actual{actual_torques}")
        # print(f"observation{st}")
        # at = self.data.ctrl
        # print(at)
        
        
        mujoco.mj_step(self.model, self.data)
        st_ = self.state.get_observation()
        # 拼接状态
        state = np.concatenate([st, self.last_action,st_,cpg_vector])
        # print(len(state))
        return state, {}
    
    def _apply_action_once(self, action):
        """把 policy 给的 (12,) action 写进 4 个 CPG，只调用一次。"""
        for i, cpg in enumerate(self.cpgs):
            mu, omega, psi = action[i*3:i*3+3]
            cpg.set_control(mu, omega, psi)
        self.action_cached = action.copy()
        return self.action_cached  




    def step(self, action):
        """
        执行一步动作，返回新状态 S、奖励、完成标志等。
        """
        # self.action_delay = []
        # self.data.ctrl[:self.n_joints] = action
        # print(f"action{action}")

        


        # t = self.t  # 当前时间
        # action_final = np.hstack([action_out,action[-1]])
        # print(len(action_final))
        # print(f"target{action_out}")
        # torques = self.data.qfrc_actuator
        # print(f"torques{torques}")
        # print("action shape: ", action.shape)
        # print("action: ", action)

        # 控制输入
        # action_out = self.controller.runStep(action)
        
        self.cpgstate = []
        joint_targets = []
        # actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        # print(actuator_names)


        if self.current_step == 0:
            action = self._apply_action_once(action)
        else:
            action = self.action_cached 
        for i, cpg in enumerate(self.cpgs):
            # mu, omega, psi = action[i*3:i*3+3]
            # cpg.set_control(mu, omega, psi)
            cpg.step(dt=0.01)
            self.cpgstate.append(cpg.get_state())
            qVal = cpg.get_joint_position(self.legs[i])  # 例如返回 [q1, q2]
            joint_targets.extend(qVal)
        cpg_vector = np.concatenate(self.cpgstate)
        joint_targets.extend([0, 0, 0, 0])
        self.data.ctrl[:self.action_dim] = joint_targets
        # print(f"joint_targets{joint_targets}")
        # print(f"joint_target{len(self.data.ctrl)}")
            
        # self.data.ctrl[:n+1] = action_final
        
        for _ in range(self.n):
            if  len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    # step_start_time = time.time()  # 每次开始模拟的实际时间
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
                self.t += self.dt    

            

            # obs_buffer = self.state.get_obs_buffer()
            # print(obs_buffer)       
            # step_start_time = time.time()  # 每次开始模拟的实际时间
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
           
        # print(f"nextobsbuffer{self.state.next_obs_buffer[0]}")  
        self.t += self.dt  
        state = self.state.get_state(action)
        state = np.concatenate([state,cpg_vector])
        # print(f"current_pos{state[0:8]}")
        
        # state = self.state.get_state(action_final)#spine
        self.base_velocity.append(state[24])
        # real_pos = state[0:8]
        # print(f"real{real_pos}")
        # diff = np.abs(action_out - real_pos)
        # print(f"diff{diff}")
        self.f_condition.append(state[12:16])
        # print(f"state{state}")
        # print(len(state))
        self.state.foot_trajectory()
        self.state.get_real_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()
        # print(f"obsbuffer{self.state.obs_buffer}") 
        self.current_step += 1
        

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
        # actual_torques = self.data.actuator_force[:self.n_joiself.legs[i]nts]
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
     
    def print_fcondition(self):
        # print(f"fcondition{self.f_condition}")
        np.save("base_velocity_no_spine.npy",self.base_velocity)
        # np.save("base_velocity_with_spine.npy",self.base_velocity)
        # np.save("foot_condition_with_spine.npy", self.f_condition)
        np.save("foot_condition_no_spine.npy", self.f_condition)
        print("foot condition have already been saved")
    
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
        # # 获取状态
        # base_position = self.state.get_body_position()
        # # base_orientation = self.data.qpos[3:7]
        # # joint_positions = self.data.qpos[7:]
        # # joint_velocities = self.data.qvel[6:]
        # # current_velocity = state[68] #no spine
        # current_pos = self.state.get_xvel()
        # current_velocity = current_pos[1]
        # # current_velocity = state[65] #spine
        # # print(current_velocity)
        # target_x = 0
        # offset_reward = -0.05 * np.abs(base_position[0] - target_x)
        # # offset_reward_oringinal = np.abs(base_position[0] - target_x)
        # # print(f"offset_reward_oringinal{offset_reward_oringinal}")

        # # (1) 前进奖励
        

        # forward_velocity = - 10 * current_velocity
        # # # 用 sigmoid 归一化并加入偏移量
        # forward_velocity_reward = 2 * (1 / (1 + np.exp(-forward_velocity))) - 1.02  
        # if forward_velocity_reward < 0:
        #     forward_velocity_reward = forward_velocity_reward/1.02
        # else:
        #     forward_velocity_reward = forward_velocity_reward/0.98
         
        #目标速度和实际速度的差值奖励
        # forward_reward = np.exp(-5*(v_target-current_velocity)**2)
        # forward_reward = np.exp(-((current_velocity-v_target)**2)/(2*sigma**2))
        
        #大动作惩罚



        # print(f"forward_velocity_reward{forward_velocity_reward}")
            
        # # === (2) 大动作惩罚 ===
        # ctrl_range = 3.0  # ±45度
        # lambda_penalty = 0.1  # 可调参数
        # action_normalized = action / ctrl_range
        # action_penalty = -lambda_penalty * np.mean(np.square(action_normalized))
        
        # # （3）姿态惩罚 pitch
        # quat = self.state.get_body_rotation_euler()
        # quat = [quat[1], quat[2], quat[3], quat[0]]
        # euler = Rotation.from_quat(quat).as_euler('xyz') 
        # print(euler)
        v_by_desired = 0.12
        v_bx_desired = 0
        w_bz_desired = 0
        v_by = self.state.get_body_velocity()[1]
        v_bx = self.state.get_body_velocity()[0]
        w_bz = self.state.get_body_angular_velocity()[2]
        w_bx,w_by = self.state.get_body_angular_velocity()[:2]
        w1,w2,w3,w4 = 1,0.42,1.39,0.59
        r_vy = np.exp(- (v_by - v_by_desired)**2 / 0.25) #1
        r_vx = np.exp(- (v_bx - v_bx_desired)**2 / 0.25) #0.5
        r_yaw = np.exp(- (w_bz - w_bz_desired)**2 / 0.25) #0.3
        r_ang_penalty = - (w_bx**2 + w_by**2) #0.2
        
        reward = w1*r_vy + w2*r_vx + w3*r_yaw + w4*r_ang_penalty

        # r_work = - np.linalg.norm(action - prev_action) #0.2


        
        # pitch = euler[0]
        # stability_penalty = -0.5 * (pitch**2)
        # stability_penalty_original = pitch**2
        # print(f"stability penalty{stability_penalty_original}")
        reward = (
                r_vy +
                r_vx +
                r_yaw +
                r_ang_penalty
            # w2 * stability_reward +
                # forward_velocity_reward +
                # forward_reward +
                # action_penalty +
                # stability_penalty +
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
         "forward_velocity_reward": r_vy,
        #  "forward_velocity_reward": forward_reward,
         "action_penalty": r_vx,
         "stability_penalty":r_yaw,
         "offset_reward": r_ang_penalty
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
        self.print_fcondition()
        self.state.save_real_trajectory()
        self.state.save_foot_trajectory()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
         
        
        




