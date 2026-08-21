# -*- coding: utf-8 -*-
"""
Rat env with spine action (方案1：spine上一帧动作放到观测向量末尾)

观测结构（新）：
    state = concat([obs_core, last_action_8, next_obs_core, spine_last_action])  # 共 73 维
其中前 72 维与旧模型完全一致（obs_core + last_action_8 + next_obs_core），
仅把脊柱上一帧动作（1 维）追加在最后，保证旧策略参数和 VecNormalize 前 72 维一一对应。

动作结构：
    action(9) = [latent_leg(8), spine(1)]
前 8 维通过 StarMapper→IK → data.ctrl[:8]
第 9 维线性放缩到脊柱关节 → data.ctrl[spine_act_index]
"""

import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize, nearest_points
from scipy.spatial import Delaunay
import matplotlib.colors as mcolors

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer

from Remostate import State
from CPGcontroller import OscillatorLeg
from LegModel.legs import LegModel
from workspace_mapping import WorkspaceDetection, alpha_shape, StarMapper, pick_incircle_center, is_valid
from energy_utils import EnergyMeter


class Go2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self,
                 render_mode: str | None = None,
                 xml_file_path: str = "/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",
                 max_steps: int = 2048,
                 spine_act_index: int = 8,
                 spine_scale: float = 1.57):
        super().__init__()

        # ====== MuJoCo ======
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.nu = int(self.model.nu)

        # Energy meter（只统计 8 个腿关节）
        qfrc_idx = [4, 3, 8, 7, 17, 16, 21, 20]
        self.energy_meter = EnergyMeter(self.model, motor_dof_idx=qfrc_idx)

        # Episode & sim
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None
        self.n = 5
        self.dt = 0.01

        # Spine config
        self.spine_act_index = spine_act_index
        self.spine_scale = spine_scale

        # State helper
        self.state = State(self.model, self.data, self.n)

        # Leg kinematics
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.fl_left  = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left  = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)
        self.legs = [self.fl_left, self.fl_right, self.hl_left, self.hl_right]

        self.n_legs = 4
        self.n_joints = 8                   # 8 个腿关节
        self.action_dim = self.n_legs*2 + 1 # 9 = 8(腿latent) + 1(spine)

        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = []

        # 记录“上一帧策略输出”（长度 9，为了更新 ctrl 和记录），
        # 但在观测里只放前 8 维 + 最后一维单独放尾部
        self.last_action_policy = np.zeros(self.action_dim, dtype=float)

        # Camera
        base_body_name = "base"
        try:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")

        # ====== Workspace mapping ======
        workspace = WorkspaceDetection(leg_params)
        grid = 400
        q_vals = np.linspace(-3.0, 3.0, grid)
        Fy_col, Fz_col = [], []
        for q1 in q_vals:
            for q2 in q_vals:
                res = workspace.angel_2_pos(q1, q2)
                if res and res[1] < 0:
                    Fy_col.append(res[0]); Fz_col.append(res[1])
        Fy_arr = np.array(Fy_col); Fz_arr = np.array(Fz_col)
        points = np.vstack((Fy_arr, Fz_arr)).T

        alpha_poly = alpha_shape(points, 100).buffer(0)
        origin = pick_incircle_center(alpha_poly)
        self.mapper = StarMapper(alpha_poly, origin=origin, num=1440, alpha_in=0.10, beta=0.05, r_thr=0.002)

        # ====== Logs ======
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = []
        self.base_velocity = []
        self.action_list = []
        self.real_pos = []

        # ====== Observation & Action spaces ======
        contact_force_low  = np.full(4, -10)
        contact_force_high = np.full(4,  10)

        # obs_core（旧环境里那一段）
        low_obs = np.concatenate((
            np.full(self.n_joints, -np.pi),  # 8
            np.full(4, -np.pi),              # 4
            contact_force_low,               # 4
            np.full(3, -np.inf),             # 3
            np.full(4, -np.pi),              # 4
            np.full(3, -np.inf),             # 3
            np.full(6, -np.inf),             # 6
        ))                                   # 总计 32
        high_obs = np.concatenate((
            np.full(self.n_joints,  np.pi),
            np.full(4,  np.pi),
            contact_force_high,
            np.full(3,  np.inf),
            np.full(4,  np.pi),
            np.full(3,  np.inf),
            np.full(6,  np.inf),
        ))  # 32

        # 观测结构（新）= obs_core(32) + last_action_8(8) + next_obs_core(32) + spine_last(1) = 73
        low_action8  = np.full(8, -1.0, dtype=np.float32)
        high_action8 = np.full(8,  1.0, dtype=np.float32)
        spine_low  = np.array([-1.0], dtype=np.float32)
        spine_high = np.array([ 1.0], dtype=np.float32)

        final_low  = np.concatenate((low_obs,  low_action8,  low_obs,  spine_low))
        final_high = np.concatenate((high_obs, high_action8, high_obs, spine_high))
        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        # 动作空间：9 维，[-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    # ---------------- Env API ----------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)

        self.energy_meter.reset()
        self.last_action_policy = np.zeros(self.action_dim, dtype=float)

        self.data.ctrl[:] = 0.0
        self.state.reset_observation()
        self.state.reset_next_observation()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # warm-up
        for _ in range(1000):
            init_ctrl = np.zeros(self.nu, dtype=float)
            init_ctrl[:8] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
            init_ctrl[self.spine_act_index] = 0.0
            self.data.ctrl[:self.nu] = init_ctrl
            mujoco.mj_step(self.model, self.data)

        # ---- 构造观测（新顺序）----
        obs_core = self.state.get_observation()  # 32
        mujoco.mj_step(self.model, self.data)
        next_obs_core = self.state.get_observation()  # 32

        last_action_8 = self.last_action_policy[:8]      # 8
        spine_last    = self.last_action_policy[8:9]     # 1

        state = np.concatenate([obs_core, last_action_8, next_obs_core, spine_last])
        info = {}
        return state.astype(np.float32), info

    def step(self, action):
        self.cpgstate = []
        E_outer = 0.0
        action = np.asarray(action, dtype=float)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action of length {self.action_dim}, got {action.shape[0]}")

        # ---- 腿：latent(8) → (Fy,Fz) → IK → ctrl[:8] ----
        for i in range(self.n_legs):
            a1 = float(action[i*2]); a2 = float(action[i*2 + 1])
            Fy, Fz = self.mapper.map_box_area_uniform(a1, a2)
            qVal = self.legs[i].pos_2_angle(np.array([Fy]), np.array([Fz]))
            action[i*2:i*2+2] = qVal

        # ---- 脊柱：线性缩放到弧度 → ctrl[spine] ----
        spine_raw = float(action[-1])  # [-1,1]
        spine_cmd = float(np.clip(spine_raw, -1.0, 1.0)) * self.spine_scale

        self.data.ctrl[:8] = action[:8]
        self.data.ctrl[self.spine_act_index] = spine_cmd

        # 记录上一帧策略动作（完整 9 维，用于下一步观测拆分）
        self.last_action_policy = action.copy()
        self.action_list.append(action.copy())
        # print(f"Step {self.current_step}: Action applied: {self.data}")
        # ---- 模拟内循环 ----
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

        # ---- 构造观测（新顺序）----
        obs_core = self.state.get_observation()
        next_obs_core = self.state.get_observation()
        last_action_8 = self.last_action_policy[:8]
        spine_last    = self.last_action_policy[8:9]
        state_vec = np.concatenate([obs_core, last_action_8, next_obs_core, spine_last]).astype(np.float32)

        # 日志（与你原先一致）
        self.base_velocity.append(state_vec[24])
        self.real_pos.append(state_vec[:8])
        self.f_condition.append(state_vec[12:16])
        self.state.foot_trajectory()
        self.state.get_real_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()

        # 奖励
        reward, reward_info = self.calculate_rewards(state_vec, action)

        # 终止
        done = bool(self.has_fallen())
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        if self.render_mode in ("window", "human") and self.viewer is not None:
            base_position = self.data.sensordata[16:19]
            self.viewer.cam.lookat[:] = base_position

        return state_vec, float(reward), done, truncated, reward_info

    # -------------- Utils & rewards --------------
    def has_fallen(self) -> bool:
        base_position = self.data.sensordata[16:19]
        return bool(base_position[2] < 0.035)

    def calculate_rewards(self, state: np.ndarray, action: np.ndarray):
        v_by_desired = -0.12
        v_bx_desired = 0
        w_bz_desired = 0
        v_by = self.state.get_body_velocity()[1]
        v_bx = self.state.get_body_velocity()[0]
        v_bz = self.state.get_body_velocity()[2]
        w_bz = self.state.get_body_angular_velocity()[2]
        w_bx, w_by = self.state.get_body_angular_velocity()[:2]
        w1, w2, w3, w4, w5 = 1.0, 0.06, 0.2, 0.117, 0.085
        r_vy = np.exp(- (v_by - v_by_desired)**2 / 0.014426)
        r_vx = np.exp(- (v_bx - v_bx_desired)**2 / 0.00130)
        r_yaw = np.exp(- (w_bz - w_bz_desired)**2 / 0.0144)
        r_vz  = - (v_bz**2)
        r_ang = - (w_bx**2 + w_by**2)
        reward = w1*r_vy + w2*r_vx + w3*r_yaw + w4*r_vz + w5*r_ang
        info = {
            "forward_velocity_reward": float(r_vy),
            "action_penalty": float(r_vx),
            "stability_penalty": float(r_yaw),
            "offset_reward": float(r_ang),
            "z_velocity_penalty": float(r_vz),
        }
        return reward, info

    # -------------- Rendering --------------
    def render(self, mode: str = "human"):
        if mode in ("human", "window"):
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                self.viewer.cam.trackbodyid = self.base_body_id
                self.viewer.cam.distance = 2.0
                self.viewer.cam.elevation = -10
                self.viewer.cam.azimuth = 90
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
            self.viewer.render()

    def close(self):
        self.state.save_real_trajectory()
        self.state.save_foot_trajectory()
        cot = self.energy_meter.cot()
        print(f"Episode CoT = {cot:.4f}")
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
