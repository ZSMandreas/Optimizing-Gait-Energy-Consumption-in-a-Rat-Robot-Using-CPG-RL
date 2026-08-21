import numpy as np
import gym
from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt
from Remostate import State
from scipy.spatial.transform import Rotation
from Controller import MouseController
from CPGcontroller import OscillatorLeg
from LegModel.forPath import LegPath
from LegModel.legs import LegModel
from workspace_sample import WorkspaceDetection,alpha_shape,is_valid,nearest_valid_point
from energy_utils import EnergyMeter
import os
os.chdir("/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")
os.chdir("/home/geriatronics/ratmujoco_ws")
    
class Go2Env(gym.Env):
    def __init__(self, render_mode=None,xml_file_path="/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",max_steps=2048):
        super(Go2Env, self).__init__()
        
        # 加载 MuJoCo 模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        qfrc_idx = [4, 3,  8, 7,  17, 16,  21, 20]
        self.energy_meter = EnergyMeter(self.model,motor_dof_idx=qfrc_idx) # test the CoT each episode
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None
        self.previous_reward = 0
        self.seed_value = None
        self.n = 5
        self.state = State(self.model, self.data, self.n)
        self.dt = 0.01
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]

        
        #能量部分
        self.E_ema = None
        self.beta = 0.99
        self.w_energy = 0.06
        self.c_max = 10.0

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
        self.legPosName = [
            ["leg_link_fl", "ankle_fl"],
            ["leg_link_fr", "ankle_fr"],
            ["leg_link_rl", "ankle_rl"],
            ["leg_link_rr", "ankle_rr"]]
        self.legRealPoint_x = [[],[],[],[]]
        self.legRealPoint_y = [[],[],[],[]]
        
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
        self.action_dim = self.n_legs*2     # 每个振荡器参数
        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = []
        self.last_action = np.zeros(self.action_dim)
        # # 所有关节名
        # joint_names = [
        #     mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        #     for j in range(self.model.njnt)
        # ]
        # for j, name in enumerate(joint_names):
        #     print(j, name)


        workspace = WorkspaceDetection(leg_params)

        grid = 400
        q_vals = np.linspace(-3.0, 3.0, grid)
        Fy_col, Fz_col = [], []

        for q1 in q_vals:
            for q2 in q_vals:
                res = workspace.angel_2_pos(q1, q2)
                if res:
                    if res[1] < 0:
                        Fy_col.append(res[0])
                        Fz_col.append(res[1])

        Fy_arr = np.array(Fy_col)
        Fz_arr = np.array(Fz_col)
        points = np.vstack((Fy_arr, Fz_arr)).T

        alpha = 100        # 越大轮廓越紧，可按需要调
        alpha_poly = alpha_shape(points, alpha)   # shapely Polygon
        
        # plt.figure(figsize=(6, 6))
        # plt.scatter(Fy_arr, Fz_arr, s=2, alpha=0.4, label='valid samples')
        # if alpha_poly.geom_type == 'Polygon':
        #     x, y = alpha_poly.exterior.xy
        #     plt.fill(x, y, 'r', alpha=0.2)
        #     plt.plot(x, y, 'r')
        # elif alpha_poly.geom_type == 'MultiPolygon':
        #     for poly in alpha_poly.geoms:
        #         x, y = poly.exterior.xy
        #         plt.fill(x, y, 'r', alpha=0.2)
        #         plt.plot(x, y, 'r')
        # plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
        # plt.axis('equal'); plt.grid(True); plt.legend()
        # plt.title("Accurate Workspace via Alpha-Shape")
        # plt.show()
    
    
        #窗口
        # self.obs_window_size = 5
        # self.action_window_size = 2
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = [] 
        self.base_velocity = []
        self.action_list = []
        self.real_pos = []

        # low_action = np.array([1.0, 0.5, -1.5]*4)
        # high_action =np.array([2.0, 4.0, 1.5]*4)
        # self.motor_ranges = [
        #     (-0.1, 0.1), (-0.1, 0.1),  # FL
        #     (-0.1, 0.1), (-0.1, 0.1),  # FR
        #     (-0.1, 0.1), (-0.1, 0.1),  # RL
        #     (-0.1, 0.1), (-0.1, 0.1),  # RR
        # ]
        # self.motor_ranges = [
        #     (-1, 1), (-1, 1),  # FL
        #     (-1, 1), (-1, 1),  # FR
        #     (-1, 1), (-1, 1),  # RL
        #     (-1, 1), (-1, 1),  # RR
        # ]
        self.motor_ranges = [
            (-0.02, 0.03), (-0.06, -0.02),  # FL
            (-0.02, 0.03), (-0.06, -0.02),  # FR
            (-0.02, 0.03), (-0.06, -0.02),  # RL
            (-0.02, 0.03), (-0.06, -0.02),  # RR
        ]
        
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
        
        contact_force_low = np.full(4, -10)  # feet condition
        contact_force_high = np.full(4, 10)
        
        # cpg_low = np.array([0.5, -5.0, 0.0, 0.0, -np.pi, -2*np.pi]*self.n_legs)
        # cpg_high = np.array([2.5, 5.0, 2*np.pi, 5.0, np.pi, 2*np.pi]*self.n_legs)
        
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
  

        # 拼接扩展后的上下界
        final_low = np.concatenate((low_obs,low_action,low_obs))
        # print(len(final_low))
        final_high = np.concatenate((high_obs,high_action,high_obs))

        # 更新观测空间
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low_action,    # [μ, ω, ψ]
            high_action,
            shape=(8,),
            dtype=np.float32
        )
        
    
    
    def reset(self, **kwargs):
        self.E_ema = None
        self.current_step = 0
        self.action_cached = None
        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)  # 设置种子
        self.energy_meter.reset() # reset the energy consumption
        self.last_action = np.zeros(self.action_dim)
        self.data.ctrl = self.last_action
        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for i in range(1000):
            self.data.ctrl = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1]
            mujoco.mj_step(self.model, self.data)
        # print(self.data.qpos)
        # print(len(self.data.qpos))
        # print(self.data.qvel)
        st = self.state.get_observation()   
        mujoco.mj_step(self.model, self.data)
        st_ = self.state.get_observation()
        # 拼接状态
        state = np.concatenate([st, self.last_action,st_])
        # print(len(state))
        return state, {}


    def step(self, action):
        """
        执行一步动作，返回新状态 S、奖励、完成标志等。
        """   
        self.cpgstate = []
        joint_targets = []
        # motor_torques = self.data.qfrc_actuator.copy()

        # print(motor_torques)
        print(f"foot_trajectory{action}") 
        # action[9:13] = 0
        workspace_penalty = 0.0
        E_outer = 0.0
        for i in range(len(self.cpgs)):
            if not is_valid(action[i*2:i*2+1],action[i*2+1:i*2+2]):
                nearest_Fy, nearest_Fz = nearest_valid_point(action[i*2:i*2+1],action[i*2+1:i*2+2])
                distance = np.sqrt((action[i*2:i*2+1] - nearest_Fy)**2 + (action[i*2+1:i*2+2] - nearest_Fz)**2)
                workspace_penalty += distance
                action[i*2:i*2+1] = nearest_Fy
                action[i*2+1:i*2+2] = nearest_Fz
            else:
                workspace_penalty += 0  
            # print(workspace_penalty)
            qVal = self.legs[i].pos_2_angle(action[i*2:i*2+1],action[i*2+1:i*2+2])
            action[i*2:i*2+2] = qVal
        self.data.ctrl[:self.action_dim] = action
        self.action_list.append(action[:8])
        
        # print(f"joint_position{action}")
        
        for _ in range(self.n):
            if  len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    # step_start_time = time.time()  # 每次开始模拟的实际时间
                    mujoco.mj_step(self.model, self.data)  # 执行一步模拟
                    _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
                    E_outer += E_inc
                    # qfrc_act = self.data.qfrc_actuator[idx]   # 执行器广义力
                    # qvel = self.data.qvel[self.n_joints]            # 对应角速度
                    # power = float(np.sum(np.abs(qfrc_act * qvel)))              # W
                    # E_outer += power * self.model.opt.timestep                  # J, 子步能量
                    # # 控制单步模拟时间
                    # step_end_time = time.time()
                    # elapsed_time = step_end_time-step_start_time
                    # print(elapsed_time)
                    # sleep_time = 0.002 - elapsed_time  # 计算补偿的睡眠时间
                    # if sleep_time > 0:
                    #     time.sleep(sleep_time)  # 使模拟的每一步耗时约为 0.002 秒
                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)
                # self.t += self.dt    

            

            # obs_buffer = self.state.get_obs_buffer()
            # print(obs_buffer)       
            # step_start_time = time.time()  # 每次开始模拟的实际时间
            mujoco.mj_step(self.model, self.data)  # 执行一步模拟
            _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
            E_outer += E_inc
            # qfrc_act = self.data.qfrc_actuator[idx]   # 执行器广义力
            # qvel = self.data.qvel[self.n_joints]            # 对应角速度
            # power = float(np.sum(np.abs(qfrc_act * qvel)))              # W
            # E_outer += power * self.model.opt.timestep                  # J, 子步能量
            # # 控制单步模拟时间
            # print(f"qfrc_act{qfrc_act}")
            # print(f"power {E_outer}")
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
        # self.t += self.dt  
        state = self.state.get_state(action)
        state = np.concatenate([state])
        
        # state = self.state.get_state(action_final)#spine
        self.base_velocity.append(state[24])
        # real_pos = state[0:8]
        self.real_pos.append(state[:8])
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
        # self.current_step += 1
        

        reward, reward_info = self.calculate_rewards(state,action)
        reward -= 1.166*workspace_penalty 
        
        #能量奖励
        # qfrc_act = self.data.qfrc_actuator[self.actuated_dof_idx]  # 执行器广义力
        # qvel     = self.data.qvel[self.actuated_dof_idx]           # 对应角速度
        # power = float(np.sum(np.abs(qfrc_act * qvel)))             # W
        # E_t   = self.dt * power                                    # J/step

        if self.E_ema is None:
            self.E_ema = E_outer                      # 冷启动，用首个观测值
        else:
            self.E_ema = self.beta*self.E_ema + (1-self.beta)*E_outer
        # 在线归一 + 压缩 + 裁剪
        E_ref = max(self.E_ema, 1e-8)
        c_t = np.log1p(E_outer / E_ref)
        c_t = float(np.clip(c_t, 0.0, self.c_max))

        # 主奖励 - 能量惩罚
        reward = reward - self.w_energy * c_t
        # print(f"workspace_penalty{workspace_penalty}")
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
        # # actual_torques = self.data.actuator_force[:self.n_joiself.legs[i]nts]
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
        np.save("joint_position_no_spine.npy",self.action_list)
        np.save("real_pos_no_spine.npy",self.real_pos)
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
        min_height = 0.035  # 动态调整阈值
        if com_z < min_height:  # 允许稍微低的质心高度
            return True
        return False


    
    def calculate_rewards(self,state,action):
        """
        计算机械鼠的奖励，包括稳定性、前进、平滑性、方向和探索奖励，支持平滑动态权重。
        """

        # # (1) 前进奖励
        v_by_desired = -0.12
        v_bx_desired = 0
        w_bz_desired = 0
        v_by = self.state.get_body_velocity()[1]
        v_bx = self.state.get_body_velocity()[0]
        v_bz = self.state.get_body_velocity()[2]
        w_bz = self.state.get_body_angular_velocity()[2]
        w_bx,w_by = self.state.get_body_angular_velocity()[:2]
        # print(w_bx, w_by)
        w1,w2,w3,w4,w5 = 1,0.06,0.2,0.117,0.085
       
        r_vy = np.exp(- (v_by - v_by_desired)**2 / 0.014426) 
        r_vx = np.exp(- (v_bx - v_bx_desired)**2 / 0.00130) 
        r_yaw = np.exp(- (w_bz - w_bz_desired)**2 / 0.0144) 
        r_vz = - v_bz**2
        # r_workspace = 
        #linear velocity in z direction 
        r_ang_penalty = - (w_bx**2 + w_by**2) 


        reward = (
                w1*r_vy +
                w2*r_vx +
                w3*r_yaw +
                w4*r_vz +
                w5*r_ang_penalty 
                
        )

        
        reward_info = {
        #  "stability_reward": stability_reward,
         "forward_velocity_reward": r_vy,
        #  "forward_velocity_reward": forward_reward,
         "action_penalty": r_vx,
         "stability_penalty":r_yaw,
         "offset_reward": r_ang_penalty,
         "z_velocity_penalty": r_vz
         }

        return reward,reward_info


    def render(self, mode="human"):
        if mode == "human":
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
        cot = self.energy_meter.cot()
        print(f"Episode CoT = {cot:.4f}")
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    

        
         
        
        




