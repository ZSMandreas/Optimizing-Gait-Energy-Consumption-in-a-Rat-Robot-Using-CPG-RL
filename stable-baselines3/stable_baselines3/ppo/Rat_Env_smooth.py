import numpy as np
import gym
from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt
from Remostate import State
from scipy.spatial.transform import Rotation
from Controller import MouseController
from CPGcontroller import OscillatorLeg
from LegModel.forPath import LegPath
from LegModel.legs import LegModel
from workspace_mapping import WorkspaceDetection, alpha_shape, StarMapper, pick_incircle_center, is_valid
from energy_utils import EnergyMeter
from shapely.geometry import Point
import matplotlib.colors as mcolors


class Go2Env(gym.Env):
    def __init__(
        self,
        render_mode=None,
        xml_file_path="/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",
        max_steps=2048,
    ):
        super(Go2Env, self).__init__()

        # ------------------- command velocities (加入观测) -------------------
        self.v_by_desired = -0.12
        self.v_bx_desired = 0.0
        self.w_bz_desired = 0.0
        # 3 维 command: [v_by_desired, v_bx_desired, w_bz_desired]
        self.command_dim = 3
        # -------------------------------------------------------------------

        # 加载 MuJoCo 模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)

        # 机器人总质量（用于 CoT）
        self.total_mass = float(np.sum(self.model.body_mass))

        # 能量 & 电机 DOF（你确认过的 8 个腿部电机 dof 索引）
        qfrc_idx = [8, 9, 12, 13, 21, 22, 25, 26]
        self.motor_dof_idx = np.array(qfrc_idx, dtype=int)

        # 电机标签：保存 CSV / 分析时用
        self.motor_labels = [
            "FL_thigh", "FL_leg",
            "FR_thigh", "FR_leg",
            "RL_thigh", "RL_leg",
            "RR_thigh", "RR_leg",
        ]

        # EnergyMeter：forward_dof 默认=1（y 方向）
        self.energy_meter = EnergyMeter(self.model, motor_dof_idx=qfrc_idx)

        # 从 DOF 索引反推每个电机在 qpos 里的索引，用于 q_err 计算
        motor_qpos_idx = []
        for dof in self.motor_dof_idx:
            jnt_id = self.model.dof_jntid[dof]        # 该 DOF 对应的 joint index
            qpos_adr = self.model.jnt_qposadr[jnt_id]  # 该 joint 在 qpos 中的起始 index
            motor_qpos_idx.append(qpos_adr)
        self.motor_qpos_idx = np.array(motor_qpos_idx, dtype=int)

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

        # CoT 参数
        self.E_ema = None
        self.beta = 0.99
        self.w_energy = 0.01
        self.c_max = 100.0

        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)

        self.legs = [
            self.fl_left,
            self.fl_right,
            self.hl_left,
            self.hl_right,
        ]
        self.legPosName = [
            ["leg_link_fl", "ankle_fl"],
            ["leg_link_fr", "ankle_fr"],
            ["leg_link_rl", "ankle_rl"],
            ["leg_link_rr", "ankle_rr"],
        ]
        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]

        # 摄像头跟踪基座
        base_body_name = "base"
        try:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")

        # 尺寸
        self.n_joints = 8
        self.n_legs = 4

        # 行为维度拆分（腿部 latent + 脊柱直接控制）
        self.leg_action_dim = self.n_legs * 2
        self.spine_action_dim = 1
        self.action_dim = self.leg_action_dim + self.spine_action_dim  # 9

        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        self.cpgstate = []
        self.last_action = np.zeros(self.action_dim)

        # === 动作平滑相关参数 ===
        # 一阶低通 / 指数平滑系数 alpha：越小越平滑，越大越跟随策略
        self.action_smooth_alpha = 0.98
        # 平滑惩罚的权重（在 reward 里惩罚相邻步的动作变化）
        self.w_action_smooth = 0.01
        # 保存上一时刻和平滑后的动作
        self.filtered_action = np.zeros(self.action_dim, dtype=float)
        self.prev_filtered_action = np.zeros(self.action_dim, dtype=float)

        # ---------- 工作空间构建 ----------
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
            r_thr=0.002,
        )

        # ===================== 采样可视化 =====================
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
        if alpha_poly.geom_type == "Polygon":
            bx, by = alpha_poly.exterior.xy
            plt.plot(bx, by, "k", lw=1.5, label="α-shape boundary")
        else:
            for poly in alpha_poly.geoms:
                bx, by = poly.exterior.xy
                plt.plot(bx, by, "k", lw=1.5, label="α-shape boundary")

        plt.imshow(
            H,
            extent=extent,
            origin="lower",
            cmap="viridis",
            norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
        )
        plt.scatter([origin[0]], [origin[1]], c="r", s=40, marker="x", label="origin")
        plt.xlabel("Fy (m)")
        plt.ylabel("Fz (m)")
        plt.title(f"Tanh-Normal Sampling + Area-Preserving + θ-CDF Mapping\n(log_std_init=0, σ={sigma:.2f})")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()
        # -------------------------------------------------------------

        # 轨迹/日志
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = []
        self.base_velocity = []
        self.action_list = []
        self.real_pos = []

        # 原始 / 平滑动作日志（方便对比）
        self.action_raw_list = []
        self.action_filtered_list = []

        # 行为范围
        self.motor_ranges = [(-1.0, 1.0) for _ in range(self.leg_action_dim)] + [(-1.57, 1.57)]
        low_action = np.array([m[0] for m in self.motor_ranges], dtype=np.float32)
        high_action = np.array([m[1] for m in self.motor_ranges], dtype=np.float32)

        # 观测上下界（原始 obs，不含 command）
        contact_force_low = np.full(4, -10)
        contact_force_high = np.full(4, 10)

        low_obs = np.concatenate(
            (
                np.full(self.n_joints, -np.pi),
                np.full(4, -np.pi),
                contact_force_low,
                np.full(3, -np.inf),
                np.full(4, -np.pi),
                np.full(3, -np.inf),
                np.full(6, -np.inf),
            )
        )
        high_obs = np.concatenate(
            (
                np.full(self.n_joints, np.pi),
                np.full(4, np.pi),
                contact_force_high,
                np.full(3, np.inf),
                np.full(4, np.pi),
                np.full(3, np.inf),
                np.full(6, np.inf),
            )
        )
        # -------- Episode 级别 reward 累加器（用于算均值） --------
        self.ep_reward_sums = {
            "r_vy": 0.0,
            "r_vx": 0.0,
            "r_yaw": 0.0,
            "r_vz": 0.0,
            "r_ang_penalty": 0.0,
            "energy_cot_step": 0.0,
            "energy_reward": 0.0,
        }
        self.ep_step_count = 0
        # --------------------------------------------------------
        # base: [obs, action, obs_next]
        final_low_base = np.concatenate((low_obs, low_action, low_obs))
        final_high_base = np.concatenate((high_obs, high_action, high_obs))

        # command = [v_by_desired, v_bx_desired, w_bz_desired]
        command_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        command_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        final_low = np.concatenate((final_low_base, command_low))
        final_high = np.concatenate((final_high_base, command_high))

        self.observation_space = spaces.Box(low=final_low, high=final_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self.E_ema = None
        self.current_step = 0
        self.action_cached = None
        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # 重置 episode 级别统计
        for k in self.ep_reward_sums:
            self.ep_reward_sums[k] = 0.0
        self.ep_step_count = 0
        # reset CoT 起点
        self.energy_meter.reset(self.data)

        self.last_action = np.zeros(self.action_dim, dtype=float)

        # 重置动作平滑状态
        self.filtered_action = np.zeros(self.action_dim, dtype=float)
        self.prev_filtered_action = np.zeros(self.action_dim, dtype=float)

        self.data.ctrl = self.last_action.copy()

        self.state.reset_observation()
        self.state.reset_next_observation()

        # 初始站立
        for _ in range(1000):
            init_ctrl = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            init_ctrl = np.concatenate([init_ctrl, [0.0]])
            self.data.ctrl[:self.action_dim] = init_ctrl
            mujoco.mj_step(self.model, self.data)

        st = self.state.get_observation()
        mujoco.mj_step(self.model, self.data)
        st_ = self.state.get_observation()

        cmd = np.array(
            [self.v_by_desired, self.v_bx_desired, self.w_bz_desired],
            dtype=np.float32,
        )

        state = np.concatenate([st, self.last_action, st_, cmd])
        return state, {}

    def step(self, action):
        """
        执行一步动作，返回新状态 S、奖励、完成标志等。
        """
        # 明确成 numpy，并 clip（双保险）
        action = np.asarray(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cpgstate = []
        joint_targets = []
        E_outer = 0.0   # 当前 RL step 的能量

        # 指数平滑（在策略输出空间做）
        self.prev_filtered_action = self.filtered_action.copy()
        # print("Action after smoothing:", self.prev_filtered_action)
        self.filtered_action = (
            self.action_smooth_alpha * action
            + (1.0 - self.action_smooth_alpha) * self.filtered_action
        )
        smoothed_action = self.filtered_action.copy()

        # 用平滑后的动作做后续 mapping
        mapping_action = smoothed_action.copy()

        # 这一 step 开始时 COM y
        y_start = float(self.data.qpos[1])

        # 这一 step 内，各电机力矩绝对值累积，用来算平均“电流”
        motor_tau_sum = np.zeros(len(self.motor_dof_idx), dtype=float)
        motor_substeps = 0

        # 腿部 latent → 关节角（使用平滑后的动作）
        for i in range(self.n_legs):
            a1 = float(smoothed_action[i * 2])
            a2 = float(smoothed_action[i * 2 + 1])
            Fy, Fz = self.mapper.map_box_area_uniform(a1, a2)
            qVal = self.legs[i].pos_2_angle(np.array([Fy]), np.array([Fz]))

            # 只改 mapping_action 前 8 维：真正下发的关节角
            mapping_action[i * 2: i * 2 + 2] = qVal

        # 下发控制：mapping_action = [腿关节角, 脊柱平滑指令]
        self.data.ctrl[:self.action_dim] = mapping_action

        # # 记录原始 / 平滑 / 映射后的动作
        # self.action_raw_list.append(action[: self.action_dim].copy())
        # self.action_filtered_list.append(smoothed_action[: self.action_dim].copy())
        # self.action_list.append(mapping_action[: self.action_dim].copy())

        # 子步仿真
        for _ in range(self.n):
            if len(self.state.obs_buffer) == 0:
                for _ in range(self.n):
                    mujoco.mj_step(self.model, self.data)
                    _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
                    E_outer += E_inc

                    tau = np.array(self.data.qfrc_actuator)[self.motor_dof_idx]
                    motor_tau_sum += np.abs(tau)
                    motor_substeps += 1

                    obs_prev = self.state.get_observation()
                    self.state.update_observations(obs_prev)

            mujoco.mj_step(self.model, self.data)
            _, E_inc = self.energy_meter.update(self.data, self.model.opt.timestep)
            E_outer += E_inc

            tau = np.array(self.data.qfrc_actuator)[self.motor_dof_idx]
            motor_tau_sum += np.abs(tau)
            motor_substeps += 1

            obs = self.state.get_observation()
            self.state.update_next_observations(obs)

        # 本 step 结束时 COM y
        y_end = float(self.data.qpos[1])
        step_dist = abs(y_end - y_start)

        # 原始状态 [obs, action, next_obs]（由 State 组装）
        base_state = self.state.get_state(action)

        cmd = np.array(
            [self.v_by_desired, self.v_bx_desired, self.w_bz_desired],
            dtype=np.float32,
        )
        # 最终给 RL 的观测 = [base_state, cmd]
        state = np.concatenate([base_state, cmd])

        # 轨迹记录
        self.base_velocity.append(state[24])
        self.real_pos.append(state[:8])
        self.f_condition.append(state[12:16])
        self.state.foot_trajectory()
        self.state.get_real_trajectory()
        self.state.obs_buffer_2 = self.state.obs_buffer.copy()
        self.state.obs_buffer = self.state.next_obs_buffer.copy()

        # 奖励（不含能量）
        reward, reward_info = self.calculate_rewards(state, action)

        # ------- 在这里累加各个分量，用于 episode 均值 -------
        self.ep_reward_sums["r_vy"] += reward_info["forward_velocity_reward"]
        self.ep_reward_sums["r_vx"] += reward_info["action_penalty"]
        self.ep_reward_sums["r_yaw"] += reward_info["stability_penalty"]
        self.ep_reward_sums["r_vz"] += reward_info["z_velocity_penalty"]
        self.ep_reward_sums["r_ang_penalty"] += reward_info["offset_reward"]
        self.ep_reward_sums["energy_cot_step"] += reward_info.get("energy_cot_step", 0.0)
        self.ep_reward_sums["energy_reward"] += reward_info.get("energy_reward", 0.0)
        self.ep_step_count += 1
        # ---------------------------------------------------

        # # === 平滑惩罚 & 平滑奖励 ===
        # action_diff = self.filtered_action - self.prev_filtered_action
        # smooth_penalty = float(np.sum(action_diff**2))
        # smooth_reward = -self.w_action_smooth * smooth_penalty  # 这一项加到总 reward 里

        # reward += smooth_reward
        # reward_info["smooth_penalty"] = smooth_penalty
        # reward_info["smooth_reward"] = smooth_reward  # <== 给 TensorBoard 用

        # ====== CoT: 用 COM y 位移算本 step 的局部 CoT ======
        if step_dist > 1e-6:
            cot_step = E_outer / (self.total_mass * 9.81 * step_dist)
            cot_step = np.clip(cot_step, 0.0, self.c_max)
        else:
            cot_step = 0.0
        reward_info["cot_step"] = float(cot_step)

        # ====== 电机平均“电流”：用平均 |tau| 近似 ======
        if motor_substeps > 0:
            avg_tau = motor_tau_sum / motor_substeps
        else:
            avg_tau = np.zeros(len(self.motor_dof_idx), dtype=float)
        reward_info["motor_avg_current"] = avg_tau.astype(float)

        # ====== q_err: 目标关节角 - 实际关节角 ======
        # 1) 目标关节角：mapping_action 前 8 维就是 4 条腿的关节角
        q_cmd_leg = np.zeros(8, dtype=float)
        for i in range(self.n_legs):
            q_cmd_leg[2 * i: 2 * i + 2] = mapping_action[2 * i: 2 * i + 2]
        reward_info["q_cmd_leg"] = q_cmd_leg.astype(float)

        # 2) 实际关节角：从 qpos 中按 motor_qpos_idx 取
        q_act_leg = self.data.qpos[self.motor_qpos_idx].copy()  # shape=(8,)
        reward_info["q_act_leg"] = q_act_leg.astype(float)

        # 3) 误差
        q_err = q_cmd_leg - q_act_leg
        reward_info["q_err"] = q_err.astype(float)

        # 同时记录原始 / 平滑的前 8 维动作
        reward_info["action_raw"] = action[:8].astype(float)
        reward_info["action_filtered"] = smoothed_action[:8].astype(float)

        done = False
        if self.has_fallen():
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        # 更新 last_action 为当前下发的 mapping_action
        self.last_action = mapping_action.copy()
        # 在 episode 结束时，把均值写进 info（reward_info）
        if done or truncated and self.ep_step_count > 0:
            reward_info["ep_r_vy_mean"] = self.ep_reward_sums["r_vy"] / self.ep_step_count
            reward_info["ep_r_vx_mean"] = self.ep_reward_sums["r_vx"] / self.ep_step_count
            reward_info["ep_r_yaw_mean"] = self.ep_reward_sums["r_yaw"] / self.ep_step_count
            reward_info["ep_r_vz_mean"] = self.ep_reward_sums["r_vz"] / self.ep_step_count
            reward_info["ep_r_ang_penalty_mean"] = (
                self.ep_reward_sums["r_ang_penalty"] / self.ep_step_count
            )
            reward_info["ep_energy_cot_step_mean"] = (
                self.ep_reward_sums["energy_cot_step"] / self.ep_step_count
            )
            reward_info["ep_energy_reward_mean"] = (
                self.ep_reward_sums["energy_reward"] / self.ep_step_count
            )
        if self.render_mode == "window" and self.viewer is not None:
            base_position = self.data.sensordata[16:19]
            self.viewer.cam.lookat[:] = base_position
        return state, reward, done, truncated, reward_info

    def print_fcondition(self):
        np.save("base_velocity_no_spine.npy", self.base_velocity)
        np.save("joint_position_no_spine.npy", self.action_list)
        np.save("real_pos_no_spine.npy", self.real_pos)
        np.save("foot_condition_no_spine.npy", self.f_condition)
        np.save("joint_position_raw_no_spine.npy", self.action_raw_list)
        np.save("joint_position_filtered_no_spine.npy", self.action_filtered_list)
        print("foot condition have already been saved")

    def _fit_state(self, obs_window):
        return np.mean(obs_window, axis=0)

    def _generate_S(self, st, at, st_next):
        return np.concatenate([st, at, st_next]).astype(np.float32)

    def _get_foot_contact_states(self):
        contact_states = []
        foot_geom_names = ["FL", "FR", "RL", "RR"]
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
        return com_z < min_height

    def calculate_rewards(self, state, action):
        """
        机械鼠奖励：速度跟踪 + 姿态稳定（不含能量惩罚）。
        """
        v_by = self.state.get_body_velocity()[1]
        v_bx = self.state.get_body_velocity()[0]
        v_bz = self.state.get_body_velocity()[2]
        w_bz = self.state.get_body_angular_velocity()[2]
        w_bx, w_by = self.state.get_body_angular_velocity()[:2]
        w1, w2, w3, w4, w5 = 1, 0.06, 0.2, 0.117, 0.085

        r_vy = np.exp(-((v_by - self.v_by_desired) ** 2) / 0.014426)
        r_vx = np.exp(-((v_bx - self.v_bx_desired) ** 2) / 0.00130)
        r_yaw = np.exp(-((w_bz - self.w_bz_desired) ** 2) / 0.0144)
        r_vz = -(v_bz**2)
        r_ang_penalty = -(w_bx**2 + w_by**2)

        reward = (
            w1 * r_vy
            + w2 * r_vx
            + w3 * r_yaw
            + w4 * r_vz
            + w5 * r_ang_penalty
        )

        reward_info = {
            "forward_velocity_reward": r_vy,
            "action_penalty": r_vx,
            "stability_penalty": r_yaw,
            "offset_reward": r_ang_penalty,
            "z_velocity_penalty": r_vz,
        }
        return reward, reward_info

    # 打印每个 DOF 的 qfrc_actuator 及 joint 名字（调试用）
    def print_qfrc_actuator_with_joint_names(self, model, data):
        for dof in range(model.nv):
            jnt_id = model.dof_jntid[dof]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            tau = data.qfrc_actuator[dof]
            print(f"dof {dof:2d} | joint {jnt_name:20s} | qfrc_actuator = {tau: .4f}")

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
        cot = self.energy_meter.cot(self.data)
        print(f"Episode CoT = {cot:.4f}")
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
