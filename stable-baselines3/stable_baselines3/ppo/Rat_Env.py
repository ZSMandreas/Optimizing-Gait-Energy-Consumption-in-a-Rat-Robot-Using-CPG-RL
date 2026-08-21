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
# Import workspace detection and mapping utilities from the revised workspace module.  
from workspace_mapping import WorkspaceDetection, alpha_shape, StarMapper, pick_incircle_center,is_valid
from energy_utils import EnergyMeter
from shapely.geometry import Point
import matplotlib.colors as mcolors
    
    
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
        self.w_energy = 0.01
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
        alpha_poly = alpha_shape(points, alpha)   # shapely Polygon or MultiPolygon

        # 为工作空间创建单射映射。在生成 α-形状后对其进行缓冲修复，再选取内部代表点作为中心。
        alpha_poly = alpha_poly.buffer(0)  # 修复几何，使其适合 StarMapper
        # origin_pt = alpha_poly.representative_point()
        # origin = (origin_pt.x, origin_pt.y)
        origin = pick_incircle_center(alpha_poly)
        # 构建 StarMapper：将 [-1,1]^2 映射到 (Fy, Fz) 工作空间。参数可适当调整。
        self.mapper = StarMapper(
            alpha_poly,
            origin=origin,
            num=1440,         # 每 0.5 度一个采样
            alpha_in=0.10,   # 内半径比例
            beta=0.05,       # 外边界裕度
            r_thr=0.002      # 有效角阈值（2~3 mm）
        )
       
        # END of workspace mapping setup
        # ===================== tanh-高斯采样并映射 =====================
        np.random.seed(42)
        N = 50000
        mu, log_std = 0.0, 0.0
        sigma = np.exp(log_std)
        eps = np.random.randn(N,2)
        x = mu + sigma*eps
        actions = np.tanh(x)  # tanh-高斯动作

        Fy_s, Fz_s = np.zeros(N), np.zeros(N)
        for i in range(N):
            Fy_s[i], Fz_s[i] = self.mapper.map_box_area_uniform(actions[i,0], actions[i,1])
            if not is_valid(Fy_s[i], Fz_s[i]):
                # 投影回最近边界点
                p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy_s[i],Fz_s[i])))
                Fy_s[i], Fz_s[i] = p_proj.x, p_proj.y
        # ===================== 绘制密度图 =====================
        # 生成二维密度直方图
        bins = 200
        H, xedges, yedges = np.histogram2d(Fy_s, Fz_s, bins=bins)
        H = H.T
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(7,7))
        # 工作空间边界
        if alpha_poly.geom_type == 'Polygon':
            bx, by = alpha_poly.exterior.xy
            plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')
        else:
            for poly in alpha_poly.geoms:
                bx, by = poly.exterior.xy
                plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')
        # 密度热图
        plt.imshow(H, extent=extent, origin='lower', cmap='viridis',
                   norm=mcolors.LogNorm(vmin=1, vmax=H.max()))
        plt.scatter([origin[0]], [origin[1]], c='r', s=40, marker='x', label='origin')
        plt.xlabel("Fy (m)")
        plt.ylabel("Fz (m)")
        plt.title(f"Tanh-Normal Sampling + Area-Preserving + θ-CDF Mapping\n(log_std_init=0, σ={sigma:.2f})")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
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

        # Define the action range for the foot workspace input.  
        # Each pair of values corresponds to (a1, a2) in the square [-1, 1]^2 for one leg.  
        # These latent parameters will be mapped into the physical workspace via StarMapper.  
        self.motor_ranges = [(-1.0, 1.0) for _ in range(self.n_legs * 2)]
        
        # Low and high bounds for the latent action box.  
        low_action = np.array([m[0] for m in self.motor_ranges])
        high_action = np.array([m[1] for m in self.motor_ranges])
        
        contact_force_low = np.full(4, -10)  # feet condition
        contact_force_high = np.full(4, 10)
        
        low_obs = np.concatenate((
            np.full(self.n_joints, -np.pi), #关节角度下界 
            np.full(4, -np.pi), # other angles
            contact_force_low,
            np.full(3, -np.inf), # base position
            np.full(4, -np.pi),#euler
            np.full(3, -np.inf),  # 基座线速度
            np.full(6, -np.inf),  # 基座加速度和角速度   
        ))

        high_obs = np.concatenate((
            np.full(self.n_joints, np.pi), #关节角度下界
            np.full(4, np.pi), # other angles
            contact_force_high,
            np.full(3, np.inf), # base position
            np.full(4, np.pi),#euler
            np.full(3, np.inf),  # 基座线速度
            np.full(6, np.inf),  # 基座加速度和角速度
        ))
  

        # 拼接扩展后的上下界
        final_low = np.concatenate((low_obs,low_action,low_obs))
        final_high = np.concatenate((high_obs,high_action,high_obs))

        # 更新观测空间
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        # The action space now represents eight latent parameters in [-1, 1] rather than foot coordinates.  
        self.action_space = spaces.Box(
            low_action,
            high_action,
            shape=(self.action_dim,),
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
        st = self.state.get_observation()   
        mujoco.mj_step(self.model, self.data)
        st_ = self.state.get_observation()
        # 拼接状态
        state = np.concatenate([st, self.last_action,st_])
        return state, {}


    def step(self, action):
        """
        执行一步动作，返回新状态 S、奖励、完成标志等。
        """   
        self.cpgstate = []
        joint_targets = []
        # rho_list = []
        # Squash the incoming action into the latent range using tanh.  
        # The raw RL action may exceed [-1, 1], so applying tanh ensures it stays within bounds.  
        # action = np.tanh(np.asarray(action, dtype=float))
        E_outer = 0.0
        for i in range(len(self.cpgs)):
            # Map the latent action (a1, a2) ∈ [-1, 1]^2 into a valid foot position (Fy, Fz) using StarMapper.  
            a1 = float(action[i*2])
            a2 = float(action[i*2 + 1])
            Fy, Fz = self.mapper.map_box_area_uniform(a1, a2)
            # print(Fy,Fz)
            # Perform inverse kinematics to convert desired foot position into joint angles.  
            qVal = self.legs[i].pos_2_angle(np.array([Fy]), np.array([Fz]))
            # Replace the latent action entries with the corresponding joint angles.  
            action[i*2:i*2+2] = qVal
            # real_pos = self.state.get_joints_pos()
            # print(action-real_pos)
            # rho_list.append(rho)
        self.data.ctrl[:self.action_dim] = action
        self.action_list.append(action[:8])
       
        
        for _ in range(self.n):
            if  len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    mujoco.mj_step(self.model, self.data)  # 执行一步模拟
                    _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
                    E_outer += E_inc
                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)
                    # rho_swing = [rho for rho, c in zip(rho_list, state[12:16]) if not c]
                    # if len(rho_swing):
                    #     w_r = getattr(self, "w_r", 0.02)  # 起步值
                    #     reward += w_r * float(np.mean(rho_swing))
            
            mujoco.mj_step(self.model, self.data)  # 执行一步模拟
            _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
            E_outer += E_inc
            obs = self.state.get_observation()  # 获取观测值
            self.state.update_next_observations(obs)
           
        state = self.state.get_state(action)
        state = np.concatenate([state])
        
        # 记录轨迹相关信息
        self.base_velocity.append(state[24])
        self.real_pos.append(state[:8])
        self.f_condition.append(state[12:16])
        # print("foot contact states:", state[12:16])
        self.state.foot_trajectory()
        self.state.get_real_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()
        
        reward, reward_info = self.calculate_rewards(state, action)
        
        # # 能量奖励
        # if self.E_ema is None:
        #     self.E_ema = E_outer                      # 冷启动，用首个观测值
        # else:
        #     self.E_ema = self.beta*self.E_ema + (1-self.beta)*E_outer
        # # 在线归一 + 压缩 + 裁剪
        # E_ref = max(self.E_ema, 1e-8)
        # c_t = np.log1p(E_outer / E_ref)
        # c_t = float(np.clip(c_t, 0.0, self.c_max))

        # # 主奖励 - 能量惩罚
        # reward = reward - self.w_energy * c_t


        # rho_swing = [rho for rho, c in zip(rho_list, state[12:16]) if not c]
        # # print("rho_swing:",rho_swing)
        # if len(rho_swing):
        #     w_r = getattr(self, "w_r", 0.02)  # 起步值
        #     reward += w_r * float(np.mean(rho_swing))
        # Workspace penalty removed; reward purely based on motion objectives and energy.
        # 判断是否完成
        done = False
        if self.has_fallen():  # 检查是否摔倒
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1  # 更新步数

        if self.render_mode == "window" and self.viewer is not None:
            # 获取基座位置
            base_position = self.data.sensordata[16:19] # 基座位置 (w, x, y, z)

            # 更新摄像机的目标位置
            self.viewer.cam.lookat[:] = base_position
        
        return state, reward, done, truncated, reward_info
     
    def print_fcondition(self):
        np.save("base_velocity_no_spine.npy",self.base_velocity)
        np.save("joint_position_no_spine.npy",self.action_list)
        np.save("real_pos_no_spine.npy",self.real_pos)
        np.save("foot_condition_no_spine.npy", self.f_condition)
        print("foot condition have already been saved")
    
    def _fit_state(self, obs_window):
        return np.mean(obs_window, axis=0)  # 对每个观测维度取均值
    
    def _generate_S(self, st, at, st_next):
        return np.concatenate([st, at, st_next]).astype(np.float32)
    
    def _get_foot_contact_states(self):
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
        base_position = self.data.sensordata[16:19]  # 基座位置 (x, y, z)
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
        w1,w2,w3,w4,w5 = 1,0.06,0.2,0.117,0.085
       
        r_vy = np.exp(- (v_by - v_by_desired)**2 / 0.014426) 
        r_vx = np.exp(- (v_bx - v_bx_desired)**2 / 0.00130) 
        r_yaw = np.exp(- (w_bz - w_bz_desired)**2 / 0.0144) 
        r_vz = - v_bz**2
        r_ang_penalty = - (w_bx**2 + w_by**2) 

        reward = (
                w1*r_vy +
                w2*r_vx +
                w3*r_yaw +
                w4*r_vz +
                w5*r_ang_penalty 
                
        )

        
        reward_info = {
         "forward_velocity_reward": r_vy,
         "action_penalty": r_vx,
         "stability_penalty":r_yaw,
         "offset_reward": r_ang_penalty,
         "z_velocity_penalty": r_vz,
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
