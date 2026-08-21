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
        xml_file_path="/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",
        max_steps=32,
    ):
        super(Go2Env, self).__init__()

        # ------------------- command velocities (加入观测) -------------------
        self.v_by_desired = -0.12
        self.v_bx_desired = 0.0
        self.w_bz_desired = 0.0
        self.command_dim = 3

        # ------------------- MuJoCo -------------------
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)

        # 质量 & 能量
        self.total_mass = float(np.sum(self.model.body_mass))
        qfrc_idx = [8, 9, 12, 13, 21, 22, 25, 26]
        self.motor_dof_idx = np.array(qfrc_idx, dtype=int)
        self.motor_labels = [
            "FL_thigh", "FL_leg",
            "FR_thigh", "FR_leg",
            "RL_thigh", "RL_leg",
            "RR_thigh", "RR_leg",
        ]
        self.energy_meter = EnergyMeter(self.model, motor_dof_idx=qfrc_idx)

        # DOF → qpos index
        motor_qpos_idx = []
        for dof in self.motor_dof_idx:
            jnt_id = self.model.dof_jntid[dof]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            motor_qpos_idx.append(qpos_adr)
        self.motor_qpos_idx = np.array(motor_qpos_idx, dtype=int)

        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None
        self.previous_reward = 0
        self.seed_value = None

        # State
        self.n = 5
        self.state = State(self.model, self.data, self.n)

        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]

        # CoT
        self.E_ema = None
        self.beta = 0.99
        self.w_energy = 0.01
        self.c_max = 100.0

        # ★ 线性前进位置奖励的权重
        self.w_pos = 1.0

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
        self.n_legs = 4
        self.n_joints = 8

        # 摄像头跟踪基座
        base_body_name = "base"
        try:
            self.base_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
            )
        except Exception as e:
            raise ValueError(f"Could not find body '{base_body_name}': {e}")

        # 动作维度：4 条腿，每条腿 2 个参数（amp, offset）
        self.leg_action_dim = self.n_legs * 2
        self.last_action = np.zeros(self.leg_action_dim, dtype=float)

        # ==== CPG 相位 ====
        self.cpg_phase = np.zeros(self.n_legs, dtype=float)
        self.cpg_freq = 2.0  # Hz
        self.cpg_phase_offset = np.array([0.0, np.pi, np.pi, 0.0], dtype=float)
        # 一个 gait 周期的子步数量
        self.gait_fraction_per_step = 1.0 / 1.0  # 你要 1/5 周期就改成 1.0/5.0
        full_cycle_substeps = 1.0 / (self.cpg_freq * self.sim_dt)
        self.time_slice_steps = max(1, int(round(self.gait_fraction_per_step * full_cycle_substeps)))

        # 动作平滑
        self.action_smooth_alpha = 0.99
        self.w_action_smooth = 0.01
        self.filtered_action = np.zeros(self.leg_action_dim, dtype=float)
        self.prev_filtered_action = np.zeros(self.leg_action_dim, dtype=float)

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
        origin = (origin[0]-0.02, origin[1]-0.005)
        self.mapper = StarMapper(
            alpha_poly,
            origin=origin,
            num=1440,
            alpha_in=0.10,
            beta=0.05,
            r_thr=0.002,
        )

        # 轨迹/日志
        self.torque_log = []
        self.speed_log = []
        self.obs_window = []
        self.f_condition = []
        self.base_velocity = []
        self.action_list = []
        self.real_pos = []

        self.action_raw_list = []
        self.action_filtered_list = []

        # 动作范围
        self.motor_ranges = [(-1.0, 1.0) for _ in range(self.leg_action_dim)]
        low_action = np.array([m[0] for m in self.motor_ranges], dtype=np.float32)
        high_action = np.array([m[1] for m in self.motor_ranges], dtype=np.float32)

        # 观测范围（根据你的 State 定义）
        low_obs = np.concatenate(
            (
                np.full(3, -np.inf, dtype=np.float32),
                np.full(4, -np.pi, dtype=np.float32),
                np.full(3, -np.inf, dtype=np.float32),
                np.full(6, -np.inf, dtype=np.float32),
            )
        )
        high_obs = np.concatenate(
            (
                np.full(3, np.inf, dtype=np.float32),
                np.full(4, np.pi, dtype=np.float32),
                np.full(3, np.inf, dtype=np.float32),
                np.full(6, np.inf, dtype=np.float32),
            )
        )
        self.obs_dim = low_obs.shape[0]

        # Episode 级 reward 累计
        self.ep_reward_sums = {
            "r_vy": 0.0,
            "r_vx": 0.0,
            "r_yaw": 0.0,
            "r_vz": 0.0,
            "r_ang_penalty": 0.0,
            "r_pos": 0.0,
            "energy_cot_step": 0.0,
            "energy_reward": 0.0,
        }
        self.ep_step_count = 0

        # 状态 = [curr_obs, curr_action, prev_obs, avg_reward, avg_vy, contact_ratio(4), command(3)]
        command_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        command_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        final_low = np.concatenate(
            (
                low_obs,
                low_action,
                low_obs,
                np.array([-np.inf], dtype=np.float32),  # avg reward
                np.array([-np.inf], dtype=np.float32),  # avg v_by
                np.zeros(4, dtype=np.float32),          # contact ratio
                command_low,
            )
        )
        final_high = np.concatenate(
            (
                high_obs,
                high_action,
                high_obs,
                np.array([np.inf], dtype=np.float32),
                np.array([np.inf], dtype=np.float32),
                np.ones(4, dtype=np.float32),
                command_high,
            )
        )

        self.observation_space = spaces.Box(
            low=final_low, high=final_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            shape=(self.leg_action_dim,),
            dtype=np.float32,
        )

        # 当前/上一 step 观测缓存
        self.curr_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self.prev_obs = np.zeros(self.obs_dim, dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, **kwargs):
        self.E_ema = None
        self.current_step = 0
        self.action_cached = None

        if "seed" in kwargs:
            seed = kwargs["seed"]
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Episode 统计
        for k in self.ep_reward_sums:
            self.ep_reward_sums[k] = 0.0
        self.ep_step_count = 0

        self.energy_meter.reset(self.data)

        # CPG 相位重置
        self.cpg_phase = self.cpg_phase_offset.copy()

        self.last_action = np.zeros(self.leg_action_dim, dtype=float)
        self.filtered_action = np.zeros(self.leg_action_dim, dtype=float)
        self.prev_filtered_action = np.zeros(self.leg_action_dim, dtype=float)

        self.data.ctrl[:] = 0.0

        self.state.reset_observation()
        self.state.reset_next_observation()

        # 初始站立
        for _ in range(1000):
            init_ctrl = np.zeros(self.leg_action_dim, dtype=float)
            self.data.ctrl[:self.leg_action_dim] = init_ctrl
            mujoco.mj_step(self.model, self.data)

        base_obs = self.state.get_observation().astype(np.float32)
        self.curr_obs = base_obs.copy()
        self.prev_obs = base_obs.copy()

        avg_reward = 0.0
        avg_vy = 0.0
        contact_ratio = np.zeros(4, dtype=np.float32)

        cmd = np.array(
            [self.v_by_desired, self.v_bx_desired, self.w_bz_desired],
            dtype=np.float32,
        )

        state = np.concatenate(
            (
                self.curr_obs,
                self.filtered_action.astype(np.float32),
                self.prev_obs,
                np.array([avg_reward], dtype=np.float32),
                np.array([avg_vy], dtype=np.float32),
                contact_ratio,
                cmd,
            )
        )
        return state, {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        """
        一个 step = 一个 gait 周期片 (time slice)，内部跑 time_slice_steps 个 mj_step。
        奖励在 time slice 结束后根据 “平均速度/角速度 + 总位移” 一次性计算。
        """
        # ---------- 1) clip + 平滑 ----------
        action = np.asarray(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.prev_filtered_action = self.filtered_action.copy()
        alpha = self.action_smooth_alpha
        self.filtered_action = alpha * self.filtered_action + (1.0 - alpha) * action
        smoothed_action = self.filtered_action.copy()

        # ---------- 2) 周期内统计量 ----------
        E_outer = 0.0

        y_start = float(self.data.qpos[1])
        y_last_sub = y_start

        motor_tau_sum = np.zeros(len(self.motor_dof_idx), dtype=float)
        motor_substeps = 0

        # 统计平均速度 / 角速度
        v_bx_sum = 0.0
        v_by_sum = 0.0
        v_bz_sum = 0.0
        w_bx_sum = 0.0
        w_by_sum = 0.0
        w_bz_sum = 0.0

        vy_sum = 0.0
        contact_counts = np.zeros(4, dtype=float)

        # 总的有符号前进位移（沿 -y 方向）
        total_dy_signed = 0.0

        omega = 2.0 * np.pi * self.cpg_freq
        mapping_action = np.zeros(self.leg_action_dim, dtype=float)

        # ---------- 2.1 substep 级日志容器 ----------
        sub_cot_list = []
        sub_motor_tau_list = []
        sub_q_cmd_list = []
        sub_q_act_list = []
        sub_q_err_list = []
        sub_action_list = []

        # ---------- 3) 子步函数 ----------
        def _update_one_substep():
            nonlocal mapping_action, E_outer, motor_tau_sum, motor_substeps
            nonlocal vy_sum, contact_counts, y_last_sub
            nonlocal v_bx_sum, v_by_sum, v_bz_sum, w_bx_sum, w_by_sum, w_bz_sum
            nonlocal total_dy_signed

            # 逐腿：CPG 相位推进 + StarMapper + IK
            for leg_idx in range(self.n_legs):
                self.cpg_phase[leg_idx] += omega * self.sim_dt
                if self.cpg_phase[leg_idx] > 2.0 * np.pi:
                    self.cpg_phase[leg_idx] -= 2.0 * np.pi

                phase = self.cpg_phase[leg_idx]

                u_theta = (phase % (2.0 * np.pi)) / (2.0 * np.pi)

                a_amp = float(smoothed_action[2 * leg_idx])
                a_bias = float(smoothed_action[2 * leg_idx + 1])

                amp = 0.5 * (a_amp + 1.0)
                offset = 0.5 * (a_bias + 1.0)

                base = 0.5 * (1.0 + np.sin(phase))
                rho = offset + (base - 0.5) * amp
                rho = np.clip(rho, 0.0, 1.0)

                a1_t = 2.0 * u_theta - 1.0
                a2_t = 2.0 * rho - 1.0

                Fy, Fz = self.mapper.map_box_area_uniform(a1_t, a2_t)
                qVal = self.legs[leg_idx].pos_2_angle(
                    np.array([Fy], dtype=float),
                    np.array([Fz], dtype=float),
                )
                mapping_action[2 * leg_idx: 2 * leg_idx + 2] = qVal

            # 下发控制
            self.data.ctrl[: self.leg_action_dim] = mapping_action

            # 一个 mj_step
            mujoco.mj_step(self.model, self.data)

            # 能量
            _, E_inc = self.energy_meter.update(self.data, self.sim_dt)
            E_outer += E_inc

            # substep 级别 CoT（近似）
            y_new = float(self.data.qpos[1])
            dy_abs = abs(y_new - y_last_sub)
            if dy_abs > 1e-6:
                cot_inst = E_inc / (self.total_mass * 9.81 * dy_abs)
                cot_inst = np.clip(cot_inst, 0.0, self.c_max)
            else:
                cot_inst = 0.0
            sub_cot_list.append(cot_inst)

            # 有符号前进位移（希望沿 -y 走，所以 y_last - y_new > 0）
            dy_signed = y_last_sub - y_new
            total_dy_signed += dy_signed
            y_last_sub = y_new

            # 电机力矩
            tau = np.array(self.data.qfrc_actuator)[self.motor_dof_idx]
            abs_tau = np.abs(tau)
            motor_tau_sum += abs_tau
            motor_substeps += 1
            sub_motor_tau_list.append(abs_tau.copy())

            # 当前关节命令 / 实际 / 误差
            q_cmd_leg = np.zeros(8, dtype=float)
            for i in range(self.n_legs):
                q_cmd_leg[2 * i: 2 * i + 2] = mapping_action[2 * i: 2 * i + 2]
            q_act_leg = self.data.qpos[self.motor_qpos_idx].copy()
            q_err = q_cmd_leg - q_act_leg

            sub_q_cmd_list.append(q_cmd_leg.copy())
            sub_q_act_list.append(q_act_leg.copy())
            sub_q_err_list.append(q_err.copy())

            # 当前 substep 使用的策略动作（平滑后）
            sub_action_list.append(smoothed_action.copy())

            # 平均前向速度 & 接触占比 & 统计平均速度 / 角速度
            body_vel = self.state.get_body_velocity()
            v_bx, v_by, v_bz = body_vel[0], body_vel[1], body_vel[2]
            v_bx_sum += v_bx
            v_by_sum += v_by
            v_bz_sum += v_bz
            vy_sum += v_by

            w = self.state.get_body_angular_velocity()
            w_bx, w_by, w_bz = w[0], w[1], w[2]
            w_bx_sum += w_bx
            w_by_sum += w_by
            w_bz_sum += w_bz

            contacts = self.state.get_feet_condition()
            contact_counts[:] += contacts.astype(float)
            self.f_condition.append(contacts.copy())

            if self.render_mode == "window":
                self.render()

        # ---------- 4) 跑完整一个 gait 周期 ----------
        for _ in range(self.time_slice_steps):
            _update_one_substep()

        # ---------- 5) 时间片平均量 ----------
        K = float(self.time_slice_steps)
        if K <= 0:
            K = 1.0

        v_bx_avg = v_bx_sum / K
        v_by_avg = v_by_sum / K
        v_bz_avg = v_bz_sum / K
        w_bx_avg = w_bx_sum / K
        w_by_avg = w_by_sum / K
        w_bz_avg = w_bz_sum / K

        avg_vy = vy_sum / K
        contact_ratio = (contact_counts / K).astype(np.float32)
        contact_ratio = np.clip(contact_ratio, 0.0, 1.0)

        y_end = float(self.data.qpos[1])
        step_dist = abs(y_end - y_start)

        # CoT
        if step_dist > 1e-6:
            cot_step = E_outer / (self.total_mass * 9.81 * step_dist)
            cot_step = np.clip(cot_step, 0.0, self.c_max)
        else:
            cot_step = 0.0

        # ★ 用平均速度/角速度计算奖励主项
        reward_main, comp_main = self.compute_reward_from_stats(
            v_bx_avg, v_by_avg, v_bz_avg, w_bx_avg, w_by_avg, w_bz_avg
        )

        # ★ 位置奖励：保持原来“均值”量级：w_pos/K * (总位移)
        # total_dy_signed ≈ y_start - y_end
        r_pos_step = self.w_pos * total_dy_signed

        reward_step = reward_main + r_pos_step

        # ---------- 6) RL 看到的观测 ----------
        base_obs_now = self.state.get_observation().astype(np.float32)
        prev_obs = self.curr_obs.copy()
        self.curr_obs = base_obs_now.copy()
        self.prev_obs = prev_obs.copy()

        cmd = np.array(
            [self.v_by_desired, self.v_bx_desired, self.w_bz_desired],
            dtype=np.float32,
        )

        state = np.concatenate(
            (
                self.curr_obs,
                action.astype(np.float32),
                self.prev_obs,
                np.array([reward_step], dtype=np.float32),
                np.array([avg_vy], dtype=np.float32),
                contact_ratio,
                cmd,
            )
        )

        # ---------- 7) reward_info ----------
        reward_info = {
            "forward_velocity_reward": comp_main["forward_velocity_reward"],
            "action_penalty": comp_main["action_penalty"],
            "stability_penalty": comp_main["stability_penalty"],
            "offset_reward": comp_main["offset_reward"],
            "z_velocity_penalty": comp_main["z_velocity_penalty"],
            "pos_reward": r_pos_step,
            "cot_step": float(cot_step),
            "avg_vy": float(avg_vy),
            "contact_ratio": contact_ratio,
        }

        # Episode 级
        self.ep_reward_sums["r_vy"] += comp_main["forward_velocity_reward"]
        self.ep_reward_sums["r_vx"] += comp_main["action_penalty"]
        self.ep_reward_sums["r_yaw"] += comp_main["stability_penalty"]
        self.ep_reward_sums["r_vz"] += comp_main["z_velocity_penalty"]
        self.ep_reward_sums["r_ang_penalty"] += comp_main["offset_reward"]
        self.ep_reward_sums["r_pos"] += r_pos_step
        self.ep_reward_sums["energy_cot_step"] += cot_step
        self.ep_reward_sums["energy_reward"] += 0.0
        self.ep_step_count += 1

        # step 级平均电机力矩
        if motor_substeps > 0:
            avg_tau = motor_tau_sum / motor_substeps
        else:
            avg_tau = np.zeros(len(self.motor_dof_idx), dtype=float)
        reward_info["motor_avg_current"] = avg_tau.astype(float)

        # step 级 q_cmd / q_act
        q_cmd_leg = np.zeros(8, dtype=float)
        for i in range(self.n_legs):
            q_cmd_leg[2 * i: 2 * i + 2] = mapping_action[2 * i: 2 * i + 2]
        reward_info["q_cmd_leg"] = q_cmd_leg.astype(float)
        q_act_leg = self.data.qpos[self.motor_qpos_idx].copy()
        reward_info["q_act_leg"] = q_act_leg.astype(float)

        reward_info["action_raw"] = action[:8].astype(float)
        reward_info["action_filtered"] = smoothed_action[:8].astype(float)

        # substep 级数组塞进 info
        reward_info["sub_cot"] = np.array(sub_cot_list, dtype=np.float32)
        reward_info["sub_motor_avg_current"] = np.array(sub_motor_tau_list, dtype=np.float32)
        reward_info["sub_q_cmd_leg"] = np.array(sub_q_cmd_list, dtype=np.float32)
        reward_info["sub_q_act_leg"] = np.array(sub_q_act_list, dtype=np.float32)
        reward_info["sub_q_err"] = np.array(sub_q_err_list, dtype=np.float32)
        reward_info["sub_action_filtered"] = np.array(sub_action_list, dtype=np.float32)

        # ---------- 8) 终止 ----------
        done = False
        if self.has_fallen():
            done = True
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        self.last_action = mapping_action.copy()

        if (done or truncated) and self.ep_step_count > 0:
            reward_info["ep_r_vy_mean"] = self.ep_reward_sums["r_vy"] / self.ep_step_count
            reward_info["ep_r_vx_mean"] = self.ep_reward_sums["r_vx"] / self.ep_step_count
            reward_info["ep_r_yaw_mean"] = self.ep_reward_sums["r_yaw"] / self.ep_step_count
            reward_info["ep_r_vz_mean"] = self.ep_reward_sums["r_vz"] / self.ep_step_count
            reward_info["ep_r_ang_penalty_mean"] = (
                self.ep_reward_sums["r_ang_penalty"] / self.ep_step_count
            )
            reward_info["ep_r_pos_mean"] = (
                self.ep_reward_sums["r_pos"] / self.ep_step_count
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

        return state, reward_step, done, truncated, reward_info

    # ------------------------------------------------------------------ 其他辅助函数保持不变
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

    # ★ 新增：用平均速度/角速度计算奖励
    def compute_reward_from_stats(self, v_bx, v_by, v_bz, w_bx, w_by, w_bz):
        w1, w2, w3, w4, w5 = 1, 0.06, 0.2, 0.117, 0.085

        r_vy = np.exp(-((v_by - self.v_by_desired) ** 2) / 0.014426)
        r_vx = np.exp(-((v_bx - self.v_bx_desired) ** 2) / 0.00130)
        r_yaw = np.exp(-((w_bz - self.w_bz_desired) ** 2) / 0.0144)
        r_vz = -(v_bz ** 2)
        r_ang_penalty = -(w_bx ** 2 + w_by ** 2)

        reward = (
            w1 * r_vy
            + w2 * r_vx
            + w3 * r_yaw
            + w4 * r_vz
            + w5 * r_ang_penalty
        )

        reward_info = {
            "forward_velocity_reward": float(r_vy),
            "action_penalty": float(r_vx),
            "stability_penalty": float(r_yaw),
            "offset_reward": float(r_ang_penalty),
            "z_velocity_penalty": float(r_vz),
        }
        return float(reward), reward_info

    # # 原来的即时版本可以保留调试用（现在 step 里不再调用）
    # def calculate_rewards(self):
    #     v = self.state.get_body_velocity()
    #     w = self.state.get_body_angular_velocity()

    #     v_bx, v_by, v_bz = v[0], v[1], v[2]
    #     w_bx, w_by, w_bz = w[0], w[1], w[2]

    #     w1, w2, w3, w4, w5 = 1, 0.06, 0.2, 0.117, 0.085

    #     r_vy = np.exp(-((v_by - self.v_by_desired) ** 2) / 0.014426)
    #     print(f"v_by: {v_by:.4f}, desired: {self.v_by_desired:.4f}, r_vy: {r_vy:.4f}")
    #     r_vx = np.exp(-((v_bx - self.v_bx_desired) ** 2) / 0.00130)
    #     r_yaw = np.exp(-((w_bz - self.w_bz_desired) ** 2) / 0.0144)
    #     r_vz = -(v_bz**2)
    #     r_ang_penalty = -(w_bx**2 + w_by**2)

    #     reward = (
    #         w1 * r_vy
    #         + w2 * r_vx
    #         + w3 * r_yaw
    #         + w4 * r_vz
    #         + w5 * r_ang_penalty
    #     )

    #     reward_info = {
    #         "forward_velocity_reward": float(r_vy),
    #         "action_penalty": float(r_vx),
    #         "stability_penalty": float(r_yaw),
    #         "offset_reward": float(r_ang_penalty),
    #         "z_velocity_penalty": float(r_vz),
    #     }
    #     return float(reward), reward_info

    def print_qfrc_actuator_with_joint_names(self, model, data):
        for dof in range(model.nv):
            jnt_id = model.dof_jntid[dof]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            tau = data.qfrc_actuator[dof]
            print(f"dof {dof:2d} | joint {jnt_name:20s} | qfrc_actuator = {tau: .4f}")

    def print_qpos_with_joint_names(self):
        for jnt_id in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            if self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                size = 7
            else:
                size = 1
            q_slice = self.data.qpos[qpos_adr: qpos_adr + size]
            print(f"joint_id {jnt_id:2d} | {jnt_name:20s} | qpos[{qpos_adr}:{qpos_adr+size}] = {q_slice}")

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
