

# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize, nearest_points
from scipy.spatial import Delaunay
from shapely.geometry import Point
import matplotlib.colors as mcolors

import gym
from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
import pandas as pd

from Remostate import State
from scipy.spatial.transform import Rotation
from Controller import MouseController
from CPGcontroller import OscillatorLeg
from LegModel.forPath import LegPath
from LegModel.legs import LegModel

# Import workspace detection and mapping utilities
from workspace_mapping import WorkspaceDetection, alpha_shape, StarMapper, pick_incircle_center, is_valid
from energy_utils import EnergyMeter

# ---------------- utility functions ----------------
def law_of_cosines_angle(la, lb, lc):
    cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb)
    if abs(cos_val) > 1:
        return -10
    return math.acos(cos_val)

def check_cross(line1, line2):
    C, D = line1
    A, E = line2
    area_CDA = (C[0]-A[0])*(D[1]-A[1]) - (C[1]-A[1])*(D[0]-A[0])
    area_CDE = (C[0]-E[0])*(D[1]-E[1]) - (C[1]-E[1])*(D[0]-E[0])
    area_AEC = (A[0]-C[0])*(E[1]-C[1]) - (A[1]-C[1])*(E[0]-C[0])
    area_AED = (A[0]-D[0])*(E[1]-D[1]) - (A[1]-D[1])*(E[0]-C[0])
    if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
        return []
    tmp = area_AEC / (area_CDE - area_CDA)
    dx = tmp * (D[0] - C[0])
    dy = tmp * (D[1] - C[1])
    return [C[0] + dx, C[1] + dy]

def alpha_shape(pts, alpha):
    if len(pts) < 4:
        return MultiPoint(list(pts)).convex_hull
    tri = Delaunay(pts)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = pts[ia], pts[ib], pts[ic]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        area = 0.5 * abs(np.cross(pb - pa, pc - pa))
        R = a*b*c / (4.0*area + 1e-12)
        if R < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_segments)
    return unary_union(list(polygonize(m)))

class Go2Env(gym.Env):
    def __init__(self, render_mode=None,
                 xml_file_path="/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",
                 max_steps=2048):
        super(Go2Env, self).__init__()

        # ----------- MuJoCo -----------
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.nu = int(self.model.nu)   # ★ 实际 actuator 数（现在为 9）

        # 能量计算沿用原 8 个腿部关节
        qfrc_idx = [4, 3, 8, 7, 17, 16, 21, 20]
        self.energy_meter = EnergyMeter(self.model, motor_dof_idx=qfrc_idx)

        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None
        self.previous_reward = 0
        self.seed_value = None
        self.n = 5
        self.dt = 0.01

        # 状态包装
        self.state = State(self.model, self.data, self.n)

        # 机械参数
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)
        self.legs = [self.fl_left, self.fl_right, self.hl_left, self.hl_right]

        self.n_joints = 8
        self.n_legs = 4
        self.action_dim = self.n_legs * 2     # ★ 策略动作仍为 8 维（四足）

        # CPG/缓存
        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = []

        # ★ ctrl 与 策略上一步动作 分离
        self.last_action_policy8 = np.zeros(self.action_dim, dtype=float)  # 用于观测拼接（保持 8 维不变）

        # 摄像机跟踪
        base_body_name = "base"
        try:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")

        # ----------- 工作空间映射（原样） -----------
        workspace = WorkspaceDetection(leg_params)
        grid = 400
        q_vals = np.linspace(-3.0, 3.0, grid)
        Fy_col, Fz_col = [], []
        for q1 in q_vals:
            for q2 in q_vals:
                res = workspace.angel_2_pos(q1, q2)
                if res and res[1] < 0:
                    Fy_col.append(res[0])
                    Fz_col.append(res[1])

        Fy_arr = np.array(Fy_col)
        Fz_arr = np.array(Fz_col)
        points = np.vstack((Fy_arr, Fz_arr)).T

        alpha = 100
        alpha_poly = alpha_shape(points, alpha)
        alpha_poly = alpha_poly.buffer(0)
        origin = pick_incircle_center(alpha_poly)

        self.mapper = StarMapper(
            alpha_poly,
            origin=origin,
            num=1440,
            alpha_in=0.10,
            beta=0.05,
            r_thr=0.002
        )

        # ----------- 采样可视化（可保留/可注释） -----------
        np.random.seed(42)
        N = 50000
        mu, log_std = 0.0, 0.0
        sigma = np.exp(log_std)
        eps = np.random.randn(N, 2)
        x = mu + sigma * eps
        actions = np.tanh(x)

        Fy_s, Fz_s = np.zeros(N), np.zeros(N)
        for i in range(N):
            Fy_s[i], Fz_s[i] = self.mapper.map_box_area_uniform(actions[i, 0], actions[i, 1])
            if not is_valid(Fy_s[i], Fz_s[i]):
                p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy_s[i], Fz_s[i])))
                Fy_s[i], Fz_s[i] = p_proj.x, p_proj.y

        bins = 200
        H, xedges, yedges = np.histogram2d(Fy_s, Fz_s, bins=bins)
        H = H.T
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(7, 7))
        if alpha_poly.geom_type == 'Polygon':
            bx, by = alpha_poly.exterior.xy
            plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')
        else:
            for poly in alpha_poly.geoms:
                bx, by = poly.exterior.xy
                plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')

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

        # ----------- 日志缓存 -----------
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = []
        self.base_velocity = []
        self.action_list = []
        self.real_pos = []

        # ----------- 观测/动作空间，保持 8 维动作不变 -----------
        contact_force_low = np.full(4, -10)
        contact_force_high = np.full(4, 10)

        low_obs = np.concatenate((
            np.full(self.n_joints, -np.pi),
            np.full(4, -np.pi),
            contact_force_low,
            np.full(3, -np.inf),
            np.full(4, -np.pi),
            np.full(3, -np.inf),
            np.full(6, -np.inf),
        ))
        high_obs = np.concatenate((
            np.full(self.n_joints, np.pi),
            np.full(4, np.pi),
            contact_force_high,
            np.full(3, np.inf),
            np.full(4, np.pi),
            np.full(3, np.inf),
            np.full(6, np.inf),
        ))

        # 动作（策略）仍为 8 维
        low_action = np.array([-1.0] * self.action_dim, dtype=np.float32)
        high_action = np.array([1.0] * self.action_dim, dtype=np.float32)

        final_low = np.concatenate((low_obs, low_action, low_obs))
        final_high = np.concatenate((high_obs, high_action, high_obs))
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    # -------------- Env API --------------
    def reset(self, **kwargs):
        self.E_ema = None
        self.current_step = 0
        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])

        self.energy_meter.reset()

        # ★ ctrl 全零（长度 = nu），策略上一步动作仍 8 维
        self.last_action_policy8 = np.zeros(self.action_dim, dtype=float)
        self.data.ctrl[:] = 0.0

        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # ★ 初始化 1000 步：前 8 通道按原来占位赋值，第 9 通道（spine）置 0
        for _ in range(1000):
            init_ctrl = np.zeros(self.nu, dtype=float)
            init_ctrl[:8] = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1]
            self.data.ctrl[:self.nu] = init_ctrl
            mujoco.mj_step(self.model, self.data)

        st = self.state.get_observation()
        mujoco.mj_step(self.model, self.data)
        st_ = self.state.get_observation()

        # 观测拼接保持与训练一致：使用 8 维 last_action_policy8
        state = np.concatenate([st, self.last_action_policy8, st_])
        return state, {}

    def step(self, action):
        """
        执行一步动作（四足 8 维），将其映射为关节角并写入前 8 个 ctrl；
        第 9 维（spine）不在此处覆盖，以便外部评测脚本叠加小幅正弦。
        """
        self.cpgstate = []
        E_outer = 0.0

        action = np.asarray(action, dtype=float)

        # ---- 四足映射：8 维 latent -> (Fy,Fz) -> IK -> 关节角 ----
        for i in range(self.n_legs):
            a1 = float(action[i * 2])
            a2 = float(action[i * 2 + 1])
            Fy, Fz = self.mapper.map_box_area_uniform(a1, a2)
            qVal = self.legs[i].pos_2_angle(np.array([Fy]), np.array([Fz]))
            action[i * 2:i * 2 + 2] = qVal

        # ★ 仅覆盖前 8 个 ctrl，spine 通道保持（供外部写入正弦）
        self.data.ctrl[:self.action_dim] = action
        self.last_action_policy8 = action.copy()
        self.action_list.append(action[:self.action_dim].copy())

        for _ in range(self.n):
            if len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    mujoco.mj_step(self.model, self.data)
                    _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
                    E_outer += E_inc
                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)

            mujoco.mj_step(self.model, self.data)
            _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
            E_outer += E_inc
            obs = self.state.get_observation()
            self.state.update_next_observations(obs)

        # 你的 State.get_state(action) 内部已按训练期格式组织观测
        state = self.state.get_state(action)
        state = np.concatenate([state])

        # 记录
        self.base_velocity.append(state[24])
        self.real_pos.append(state[:8])
        self.f_condition.append(state[12:16])
        self.state.foot_trajectory()
        self.state.get_real_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()

        reward, reward_info = self.calculate_rewards(state, action)

        done = False
        if self.has_fallen():
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        if self.render_mode == "window" and self.viewer is not None:
            base_position = self.data.sensordata[16:19]
            self.viewer.cam.lookat[:] = base_position

        return state, reward, done, truncated, reward_info

    # -------------- Utils / Rewards --------------
    def print_fcondition(self):
        np.save("base_velocity_no_spine.npy", self.base_velocity)
        np.save("joint_position_no_spine.npy", self.action_list)
        np.save("real_pos_no_spine.npy", self.real_pos)
        np.save("foot_condition_no_spine.npy", self.f_condition)
        print("foot condition have already been saved")

    def _fit_state(self, obs_window):
        return np.mean(obs_window, axis=0)

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
        return np.array(contact_states, dtype=np.float32)

    def has_fallen(self):
        base_position = self.data.sensordata[16:19]
        com_z = base_position[2]
        min_height = 0.035
        return bool(com_z < min_height)

    def calculate_rewards(self, state, action):
        v_by_desired = -0.12
        v_bx_desired = 0
        w_bz_desired = 0
        v_by = self.state.get_body_velocity()[1]
        v_bx = self.state.get_body_velocity()[0]
        v_bz = self.state.get_body_velocity()[2]
        w_bz = self.state.get_body_angular_velocity()[2]
        w_bx, w_by = self.state.get_body_angular_velocity()[:2]
        w1, w2, w3, w4, w5 = 1, 0.06, 0.2, 0.117, 0.085

        r_vy = np.exp(- (v_by - v_by_desired)**2 / 0.014426)
        r_vx = np.exp(- (v_bx - v_bx_desired)**2 / 0.00130)
        r_yaw = np.exp(- (w_bz - w_bz_desired)**2 / 0.0144)
        r_vz = - v_bz**2
        r_ang_penalty = - (w_bx**2 + w_by**2)

        reward = (
            w1 * r_vy +
            w2 * r_vx +
            w3 * r_yaw +
            w4 * r_vz +
            w5 * r_ang_penalty
        )

        reward_info = {
            "forward_velocity_reward": r_vy,
            "action_penalty": r_vx,
            "stability_penalty": r_yaw,
            "offset_reward": r_ang_penalty,
            "z_velocity_penalty": r_vz,
        }
        return reward, reward_info

    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                self.viewer.cam.trackbodyid = self.base_body_id
                self.viewer.cam.distance = 2.0
                self.viewer.cam.elevation = -10
                self.viewer.cam.azimuth = 90
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
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
