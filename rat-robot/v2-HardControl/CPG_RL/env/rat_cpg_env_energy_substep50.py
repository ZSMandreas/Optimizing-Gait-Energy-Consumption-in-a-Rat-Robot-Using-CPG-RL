# rat_cpg_env_energy_substep50.py
# 复制自 rat_cpg_env_energy.py，修改为每 env.step() 执行 K=50 次 mj_step（substep）。
# 1) ✅ action 8 维：f(4)+mu(4)
# 2) ✅ contact 使用 sensordata 索引：12-15 对应 fl/fr/rl/rr touch
# 3) ✅ CPG 相位：每个 sim step 更新一次（foot_path.step() 在 K 次循环内每次调用）；CPG 参数 (f, mu) 每 env step 更新一次。
# 4) ✅ 每步 physics：每个 sim step 先更新相位、再 IK 得到 joint_targets、再 mj_step；能量/距离等按 50 步累加，obs/reward 取最终状态。
#
# 做法B（action_update_interval > 1）：仅每 N 个 env step 才用新 action 更新 CPG，中间步沿用上一拍的 (f,μ)；
# 训练时需配合 ActionRepeatModelWrapper，使 rollout 中存的动作与实际执行一致；info["action_used"] 为本步实际使用的动作。
# 参数对应关系：
# - action_update_interval 默认 1：每 1 个 env step 更新一次 CPG 参数 (f, mu)；>1 时每 N 步更新一次。
# - max_episode_steps：truncate 按 sim_step_counter（sim steps）。
# - 步态周期：按 sim step 数；例如 fre=0.5 Hz、dt=0.002 时，一周期 = 1000 sim steps = 20 env steps。
# - max_episode_steps：按 sim steps 截断；默认 2050 = 41 env steps × 50 substeps，与 rollout 一致。
# - CSV 记录：按 sim step 写入（每 env step 写 50 行）；episode summary 按 episode 写一行。

from __future__ import annotations

import math
import csv
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from energy_tank import EnergyTankJointSpace

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore


# ---------------------------------------------------------------------
# 1) IK (Scheme B: math-consistent + explicit failure via None)
# ---------------------------------------------------------------------

class LegModel:
    """
    IK for planar leg (y-z plane) based on your LegModel.

    Scheme B:
    - Use same geometry relations as your original
    - Add minimal numerical guards:
        * AF near zero
        * acos arg clamp
        * LawOfCosines denominator guard and range check
    - If unreachable -> return None
    """

    def __init__(self, leg_params: List[float]):
        self.len = leg_params
        self.By = 0.0
        self.Bz = float(self.len[1])

        # kept from your code (not used directly here)
        self.limit_CBz = self.LawOfCosines_angle(self.len[2], self.len[1], 0.0075)
        self.limit_DCB = self.LawOfCosines_angle(0.012735, self.len[2], 0.002)
        self.limit_AEF = self.LawOfCosines_angle(0.01025, 0.01025, 0.0042)

    @staticmethod
    def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def pos_2_angle(self, Fy: float, Fz: float) -> Optional[Tuple[float, float]]:
        PI = math.pi

        AF = math.sqrt(Fy * Fy + Fz * Fz)
        if AF < 1e-9:
            return None

        a_FAy = math.acos(self._clamp(Fy / AF))
        if Fz < 0:
            a_FAy = -a_FAy

        a_FAE = self.LawOfCosines_angle(AF, self.len[0], self.len[5])
        if a_FAE is None:
            return None

        q1 = a_FAE + a_FAy

        Ey = self.len[0] * math.cos(q1)
        Ez = self.len[0] * math.sin(q1)

        DE_FE = self.len[4] / self.len[5]
        Dy = Ey + DE_FE * (Ey - Fy)
        Dz = Ez + DE_FE * (Ez - Fz)

        AD = math.sqrt(Dy * Dy + Dz * Dz)
        BD = math.sqrt((self.By - Dy) * (self.By - Dy) + (self.Bz - Dz) * (self.Bz - Dz))

        a_ABD = self.LawOfCosines_angle(self.len[1], BD, AD)
        if a_ABD is None:
            return None

        if Dy < 0:
            if Dz < self.Bz:
                a_ABD = -a_ABD
            else:
                a_ABD = 2 * PI - a_ABD

        a_DBC = self.LawOfCosines_angle(self.len[2], BD, self.len[3])
        if a_DBC is None:
            return None

        a_ABC = a_ABD + a_DBC
        q2 = a_ABC - PI
        return float(q1), float(q2)

    def LawOfCosines_angle(self, la: float, lb: float, lc: float) -> Optional[float]:
        denom = 2.0 * la * lb
        if denom < 1e-12:
            return None
        cosv = (la * la + lb * lb - lc * lc) / denom
        if abs(cosv) > 1.0:
            return None
        return float(math.acos(self._clamp(cosv)))


# ---------------------------------------------------------------------
# 2) CPG network (per-leg f, per-leg mu, FIXED trot offsets)
# ---------------------------------------------------------------------

class CPGNetwork4:
    """
    4-leg coupled oscillator network (polar form):

      r_dot     = alpha*(mu_i - r_i^2)*r_i
      theta_dot = omega_i + sum_j K_ij * sin(theta_j - theta_i - phi_ij)

    leg order: 0 FL, 1 FR, 2 RL, 3 RR
    """

    def __init__(
        self,
        fre_cyc: float | np.ndarray,
        dt: float,
        desired_offsets: List[float],
        alpha: float = 30.0,
        mu: float | np.ndarray = 1.0,
        k: float = 8.0,
    ) -> None:
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.N = 4

        self.desired = np.array(desired_offsets, dtype=float).reshape(-1) % (2.0 * math.pi)
        if self.desired.shape[0] != 4:
            raise ValueError("desired_offsets must have length 4")
        self._recompute_phi()

        self.set_k(k)
        self.set_freq(fre_cyc)
        self.set_mu(mu)

        self.theta = self.desired.copy()
        self.r = np.sqrt(np.maximum(self.mu_vec, 1e-12)).astype(float)

    def _recompute_phi(self) -> None:
        self.phi = (self.desired.reshape(1, -1) - self.desired.reshape(-1, 1))

    def set_k(self, k: float) -> None:
        self.k = float(k)
        self.K = np.ones((self.N, self.N), dtype=float) * self.k
        np.fill_diagonal(self.K, 0.0)

    def set_freq(self, fre_cyc: float | np.ndarray) -> None:
        if isinstance(fre_cyc, np.ndarray):
            f = fre_cyc.astype(float).reshape(-1)
            if f.size != 4:
                raise ValueError("fre_cyc array must have length 4")
            self.fre_cyc = f.copy()
        else:
            self.fre_cyc = np.ones(4, dtype=float) * float(fre_cyc)
        self.omega_vec = 2.0 * math.pi * self.fre_cyc

    def set_mu(self, mu: float | np.ndarray) -> None:
        if isinstance(mu, np.ndarray):
            m = mu.astype(float).reshape(-1)
            if m.size != 4:
                raise ValueError("mu array must have length 4")
            self.mu_vec = m.copy()
        else:
            self.mu_vec = np.ones(4, dtype=float) * float(mu)

    def set_desired_offsets(self, desired_offsets: np.ndarray) -> None:
        d = np.asarray(desired_offsets, dtype=float).reshape(-1)
        if d.size != 4:
            raise ValueError("desired_offsets must have length 4")
        self.desired = d % (2.0 * math.pi)
        self._recompute_phi()

    def reset(self, init_thetas: List[float] | None = None, init_rs: List[float] | None = None) -> None:
        if init_thetas is None:
            init_thetas = self.desired.copy().tolist()
        if init_rs is None:
            init_rs = np.sqrt(np.maximum(self.mu_vec, 1e-12)).tolist()
        self.theta = np.array(init_thetas, dtype=float) % (2.0 * math.pi)
        self.r = np.array(init_rs, dtype=float)

    def step(self) -> None:
        r = self.r
        th = self.theta

        mu = self.mu_vec
        r_dot = self.alpha * (mu - r * r) * r

        th_i = th.reshape(-1, 1)
        th_j = th.reshape(1, -1)
        diff = th_j - th_i
        coupling = np.sum(self.K * np.sin(diff - self.phi), axis=1)

        th_dot = self.omega_vec + coupling

        self.r = r + r_dot * self.dt
        self.theta = (th + th_dot * self.dt) % (2.0 * math.pi)

    def get_phase(self, leg_id: int) -> float:
        return float(self.theta[int(leg_id)])

    def get_amp(self, leg_id: int) -> float:
        return float(self.r[int(leg_id)])


# ---------------------------------------------------------------------
# 3) Foot trajectory generator (fixed shape para_FU/FD/HU/HD, scaled by amp=sqrt(mu))
# ---------------------------------------------------------------------

class FootPathFixed:
    """
    Fixed oval parameters (your para_FU/FD/HU/HD):
      Each: [[y0, z0], [a, b]]
    For each leg:
      choose up/down by phase split
      Fy = y0 + (a * amp) * cos(phi)
      Fz = z0 + (b * amp) * sin(phi)
    amp = clip( sqrt(mu_leg), AMP_CLIP )

    ✅ Offsets are fixed to base_desired_offsets (selected gait template). No learned offsets.
    """

    def __init__(
        self,
        dt: float,
        base_desired_offsets: List[float],
        k_coupling: float = 3.0,
        alpha: float = 30.0,
        z_asym_alpha: float = 0.0,
        z_amp_scale: float = 1.0,
    ) -> None:
        self.dt = float(dt)

        self.para_FU = [[-0.00, -0.045], [0.03, 0.01]]
        self.para_FD = [[-0.00, -0.045], [0.03, 0.005]]
        self.para_HU = [[-0.005, -0.05], [0.03, 0.01]]
        self.para_HD = [[-0.005, -0.05], [0.03, 0.005]]

        self.halfPeriod = 1.0
        self.AMP_CLIP = (0.00, 2.0)
        self.z_asym_alpha = float(np.clip(z_asym_alpha, 0.0, 1.0))
        self.z_amp_scale = float(max(z_amp_scale, 0.0))

        self.base_desired_offsets = np.array(base_desired_offsets, dtype=float).reshape(-1)
        if self.base_desired_offsets.size != 4:
            raise ValueError("base_desired_offsets must be length 4")

        self.cpg = CPGNetwork4(
            fre_cyc=1.2,
            dt=self.dt,
            desired_offsets=self.base_desired_offsets.tolist(),
            alpha=alpha,
            mu=np.ones(4, dtype=float),
            k=k_coupling,
        )

    def reset(self, phases: List[float] | None = None) -> None:
        self.cpg.reset(init_thetas=phases)

    def step(self) -> None:
        self.cpg.step()

    def set_controls(self, f4: np.ndarray, mu4: np.ndarray) -> None:
        self.cpg.set_freq(np.asarray(f4, dtype=float))
        self.cpg.set_mu(np.asarray(mu4, dtype=float))
        # keep desired_offsets fixed (selected gait template)
        self.cpg.set_desired_offsets(self.base_desired_offsets)

    def set_shape_uniform(self, a: float, b: float) -> None:
        a_val = float(a)
        b_val = float(b)
        self.para_FU[1][0] = a_val
        self.para_FD[1][0] = a_val
        self.para_HU[1][0] = a_val
        self.para_HD[1][0] = a_val
        self.para_FU[1][1] = b_val
        self.para_FD[1][1] = b_val
        self.para_HU[1][1] = b_val
        self.para_HD[1][1] = b_val

    def get_leg_phase(self, leg_id: int) -> float:
        return self.cpg.get_phase(leg_id)

    def get_leg_amp(self, leg_id: int) -> float:
        return self.cpg.get_amp(leg_id)

    def in_swing_phase(self, phi: float) -> bool:
        # Existing phase definition in this project:
        # swing/up branch is phi in [0, pi), stance/down branch is [pi, 2pi).
        return bool(phi < float(self.halfPeriod) * math.pi)

    def normalize_to_swing(self, phi: float) -> float:
        # Keep current CPG phase convention and map swing phase to [0, pi].
        if self.in_swing_phase(phi):
            return float(phi)
        return float(phi - math.pi)

    def compute_foot_z(self, phi: float, z0: float, b0: float, amp: float) -> float:
        # Baseline is exactly the original mapping when z_asym_alpha=0 and z_amp_scale=1.
        a = self.z_asym_alpha
        k_z = self.z_amp_scale
        A_z = b0 * amp
        A_eff = A_z * k_z
        if not self.in_swing_phase(phi):
            return float(z0 + A_eff * math.sin(phi))
        theta_sw = self.normalize_to_swing(phi)
        if theta_sw <= (math.pi / 2.0):
            z_rel = A_eff * math.sin(theta_sw)
        else:
            z_std = A_eff * math.sin(theta_sw)
            z_cosine = A_eff * (math.cos(theta_sw - math.pi / 2.0) ** 2)
            z_rel = (1.0 - a) * z_std + a * z_cosine
        return float(z0 + z_rel)

    def _base_params_for_leg(self, leg_id: int, phi: float) -> Tuple[float, float, float, float]:
        is_front = (leg_id < 2)
        is_up = (phi < float(self.halfPeriod) * math.pi)

        if is_front:
            pp = self.para_FU if is_up else self.para_FD
        else:
            pp = self.para_HU if is_up else self.para_HD

        (y0, z0) = pp[0]
        (a0, b0) = pp[1]
        return float(y0), float(z0), float(a0), float(b0)

    def get_foot_target(self, leg_id: int) -> Tuple[float, float]:
        phi = self.cpg.get_phase(leg_id)

        mu_leg = float(self.cpg.mu_vec[int(leg_id)])
        amp = math.sqrt(max(mu_leg, 1e-12))
        amp = float(np.clip(amp, self.AMP_CLIP[0], self.AMP_CLIP[1]))

        y0, z0, a0, b0 = self._base_params_for_leg(leg_id, phi)
        a = a0 * amp
        b = b0 * amp

        Fy = y0 + a * math.cos(phi)
        Fz = self.compute_foot_z(phi=phi, z0=z0, b0=b0, amp=amp)
        return float(Fy), float(Fz)


# ---------------------------------------------------------------------
# 4) Gym Env
# ---------------------------------------------------------------------

class RatCPGEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    FOOT_SITES = ["ankle_fl", "ankle_fr", "ankle_rl", "ankle_rr"]

    # action ranges (offset 保留配置但不再用于 action)
    ACTION_RANGES = {
        "f": (0.3, 2.5),                       # per-leg (increased f_hi)
        "offset": (-0.2 * math.pi, 0.2 * math.pi),  # kept (unused)
        "mu": (0.2, 1.0),                       # per-leg
    }

    # sensordata indices (as you specified)
    SENSOR_TOUCH_SLICE = slice(12, 16)  # fl, fr, rl, rr
    # IMU-based sensors (overall layout per dynamic_4l.xml comment):
    # 0-7:  leg servo angles (8)
    # 8-11: tail, neck, head, spine servo angles (4)
    # 12-15: foot touch sensors (4)
    # 16-18: com_pos (framepos, 3)
    # 19-22: com_quat (framequat, 4)
    # 23-25: com_vel (framelinvel, 3)
    # 26-28: imu_acc (accelerometer, 3)
    # 29-31: imu_gyro (gyro, 3)
    IMU_POS_SLICE = slice(16, 19)
    IMU_VEL_SLICE = slice(23, 26)
    IMU_GYRO_SLICE = slice(29, 32)

    def __init__(
        self,
        model_path: str = "/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml",
        render_mode: str | None = None,
        target_speed: float = 0.12,  # forward along -Y (m/s), match plot_rewards.py
        gait_template: str = "trot",
        action_update_interval: int = 1,
        smoothing_alpha: float = 1,

        # Reward weights
        vel_weight: float = 1.0,
        pose_weight: float = 0.2,
        x_weight: float = 0.1,
        smooth_weight: float = 0.02,
        ik_fail_weight: float = 0.05,
        torque_smooth_weight: float = 0.01,
        sat_weight: float = 2.0,
        sat_threshold: float = 0.90,
        joint_energy_weight: float = 0.03,
        neg_power_alpha: float = 0.5,
        impact_weight: float = 0.01,
        body_height_weight: float = 0.02,
        body_vz_weight: float = 0.02,

        # CoT reward
        cot_weight: float = 0.0,
        cot_clip: float = 50.0,
        cot_min_dist: float = 0.05,

        # kept for compatibility/logging
        energy_weight: float = 0.0,

        # Tolerances for normalization
        x_tol: float = 0.03,
        roll_tol_deg: float = 20.0,
        pitch_tol_deg: float = 20.0,

        # Velocity reward shape
        vel_reward_denom: float = 0.0144,
        # Gaussian width for r_vel (m/s). Default 0.08 matches v_cmd=0.12 training.
        # For other target speeds use 0.08 * (target_speed / 0.12) for equal relative width.
        vel_reward_sigma: float = 0.08,
        # Lateral (X) velocity penalty width for r_xvel. None => 0.08 * (target_speed / 0.12).
        xvel_reward_sigma: float | None = None,
        # r_vel = (g_v - g0) / (1 - g0): when |target_speed| -> 0, (1-g0) -> 0 and r_vel explodes.
        # Use max(|target_speed|, vel_reward_ref_floor) only for computing g0 (not for vel_err).
        vel_reward_ref_floor: float = 0.04,
        vel_reward_den_floor: float = 1e-4,

        # Randomize CPG initial phase
        randomize_cpg_phase: bool = True,
        # 与 sim_test 对齐：True = 用 CPG 相位定义 contact/stance（φ≥π 为支撑），False = 用 sensordata 物理接触
        use_phase_for_contact: bool = False,
        # True = 忽略 action，始终用 f4=[0.5]*4、mu4=[1.0]*4（与 sim_test 一致，用于验证 env 能否直线前进）
        fixed_cpg: bool = False,
        # CPG foot-z shaping (evaluation-only knobs; defaults keep baseline behavior)
        cpg_z_asym_alpha: float = 0.0,
        cpg_z_amp_scale: float = 1.0,
        # True = IK 输出经 energy tank（关节空间虚拟阻抗 + 能量罐 + 导纳）得到 q_ref 再写 ctrl，限制正向做功
        use_energy_tank: bool = False,
        max_episode_steps: int = 2050,  # 41 env steps × 50 substeps，与 rollout n_steps=41 对应

        # substeps per env step (K=50)；CPG 相位每 sim step 更新，CPG 参数 (f, mu) 每 env step 更新
        n_substeps: int = 50,
        enable_extended_logging: bool = True,
        # CSV：每 sim step 一行（每 env step 50 行）
        enable_csv_log: bool = False,
        csv_path: str = "rat_debug.csv",
        auto_timestamp_csv: bool = True,
        print_sim_step_torque: bool = False,
        print_sim_step_joint_angles: bool = False,
        # ✅ reset 时是否对 CPG 参数做随机扰动（课程式用）
        randomize_cpg_params: bool = False,
        reset_f_sigma: float = 0.05,          # Hz, 围绕中点扰动（与 ACTION_RANGES["f"]=(0.3,1) 匹配）
        reset_mu_sigma: float = 0.05,         # mu扰动
        reset_off_sigma_pi: float = 0.05,     # kept (unused)
        # If set, reset CPG f4 to this (Hz) instead of mid-range ~1.4 — helps low-speed M3-like gait.
        init_f4_hz: float | None = None,

        # episode summary logging
        enable_episode_summary: bool = True,
        episode_summary_path: str = "rat_episode_summary.csv",
        auto_timestamp_episode_summary: bool = False,
        # B (ET): per-joint K/D vectors + optional trot-group scale on tank step (defaults off → old scalar tank)
        et_vector_kd: bool = False,
        et_group_scale: bool = False,
        et_w_mod_min: float = 0.85,
        et_w_mod_max: float = 1.15,
        et_k_hip: float | None = None,
        et_k_knee: float | None = None,
        et_d_hip: float | None = None,
        et_d_knee: float | None = None,
        # C (explicit joint coupling in ET virtual torque; defaults off)
        et_joint_coupling: bool = False,
        et_couple_phase_gate: bool = True,
        et_couple_k_ratio: float = 0.10,
        et_couple_d_ratio: float = 0.10,
        et_diag_leg_coupling: bool = False,
        et_diag_leg_k_ratio: float = 0.05,
        et_diag_leg_d_ratio: float = 0.05,
        # EnergyTankJointSpace scalars (only when use_energy_tank=True); larger E_max / A reduces alpha clipping / lag
        et_tank_E0: float = 1.0,
        et_tank_E_max: float = 3.0,
        et_tank_A: float = 100.0,
        et_tank_eta: float = 0.8,
    ) -> None:
        super().__init__()
        if mujoco is None:  # pragma: no cover
            raise RuntimeError("The mujoco package is required but is not installed.")

        self.model_path = model_path
        self.render_mode = render_mode

        self.target_speed = float(target_speed)
        self.action_update_interval = int(action_update_interval)
        self.smoothing_alpha = float(smoothing_alpha)

        # reward params
        self.vel_weight = float(vel_weight)
        self.pose_weight = float(pose_weight)
        self.x_weight = float(x_weight)
        self.torque_smooth_weight = float(torque_smooth_weight)
        self.smooth_weight = float(smooth_weight)
        self.ik_fail_weight = float(ik_fail_weight)
        self.sat_weight = float(max(sat_weight, 0.0))
        self.sat_threshold = float(np.clip(sat_threshold, 0.0, 1.0))
        self.joint_energy_weight = float(max(joint_energy_weight, 0.0))
        self.neg_power_alpha = float(max(neg_power_alpha, 0.0))
        self.impact_weight = float(max(impact_weight, 0.0))
        self.body_height_weight = float(max(body_height_weight, 0.0))
        self.body_vz_weight = float(max(body_vz_weight, 0.0))

        self.cot_weight = float(cot_weight)
        self.cot_clip = float(max(cot_clip, 1e-9))
        self.cot_min_dist = float(max(cot_min_dist, 0.0))

        self.energy_weight = float(energy_weight)

        self.x_tol = float(max(x_tol, 1e-6))
        self.roll_tol = float(max(math.radians(float(roll_tol_deg)), 1e-6))
        self.pitch_tol = float(max(math.radians(float(pitch_tol_deg)), 1e-6))
        self.vel_reward_denom = float(max(vel_reward_denom, 1e-12))
        self.vel_reward_sigma = float(max(vel_reward_sigma, 1e-6))
        if xvel_reward_sigma is None:
            self.xvel_reward_sigma = 0.08 * (abs(self.target_speed) / 0.12)
        else:
            self.xvel_reward_sigma = float(max(xvel_reward_sigma, 1e-6))
        self.vel_reward_ref_floor = float(max(vel_reward_ref_floor, 0.0))
        self.vel_reward_den_floor = float(max(vel_reward_den_floor, 1e-12))

        self.randomize_cpg_phase = bool(randomize_cpg_phase)
        self.use_phase_for_contact = bool(use_phase_for_contact)
        self.fixed_cpg = bool(fixed_cpg)
        self.cpg_z_asym_alpha = float(np.clip(cpg_z_asym_alpha, 0.0, 1.0))
        self.cpg_z_amp_scale = float(max(cpg_z_amp_scale, 0.0))
        self.use_energy_tank = bool(use_energy_tank)
        self.et_vector_kd = bool(et_vector_kd)
        self.et_group_scale = bool(et_group_scale)
        self.et_w_mod_min = float(et_w_mod_min)
        self.et_w_mod_max = float(et_w_mod_max)
        self.et_joint_coupling = bool(et_joint_coupling)
        self.et_couple_phase_gate = bool(et_couple_phase_gate)
        self.et_couple_k_ratio = float(et_couple_k_ratio)
        self.et_couple_d_ratio = float(et_couple_d_ratio)
        self.et_diag_leg_coupling = bool(et_diag_leg_coupling)
        self.et_diag_leg_k_ratio = float(et_diag_leg_k_ratio)
        self.et_diag_leg_d_ratio = float(et_diag_leg_d_ratio)
        e0 = max(float(et_tank_E0), 1e-9)
        emax = max(float(et_tank_E_max), e0)
        self.et_tank_E0 = e0
        self.et_tank_E_max = emax
        self.et_tank_A = max(float(et_tank_A), 1e-12)
        self.et_tank_eta = float(np.clip(float(et_tank_eta), 1e-6, 1.0))
        self.max_episode_steps = int(max_episode_steps)
        self.n_substeps = max(1, int(n_substeps))
        self.enable_extended_logging = bool(enable_extended_logging)
        self.randomize_cpg_params = bool(randomize_cpg_params)
        self.reset_f_sigma = float(reset_f_sigma)
        self.reset_mu_sigma = float(reset_mu_sigma)
        self.init_f4_hz = None if init_f4_hz is None else float(init_f4_hz)
        self.reset_off_sigma_pi = float(reset_off_sigma_pi)  # kept (unused)

        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.dt_sim = float(self.dt)

        # diagnostics
        self.total_mass = float(np.sum(self.model.body_mass))
        self.g = float(abs(self.model.opt.gravity[2])) if abs(self.model.opt.gravity[2]) > 1e-6 else 9.81

        # base desired offsets (FIXED template selected by gait_template)
        self.gait_template = str(gait_template).strip().lower()
        self.base_desired_offsets = self._resolve_gait_template_offsets(self.gait_template)

        # fixed-shape foot path + CPG (k fixed)
        self.foot_path = FootPathFixed(
            dt=self.dt,
            base_desired_offsets=self.base_desired_offsets.tolist(),
            k_coupling=3.0,
            alpha=30.0,
            z_asym_alpha=self.cpg_z_asym_alpha,
            z_amp_scale=self.cpg_z_amp_scale,
        )

        # IK models
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.leg_models = [LegModel(leg_params) for _ in range(4)]

        # Turn angles
        self.turn_F = 0.0 * math.pi / 180.0
        self.turn_H = 12.0 * math.pi / 180.0

        # Controlled joints
        self.controlled_joint_names: List[Tuple[str, str]] = [
            ("thigh_joint_fl", "leg_joint_fl"),
            ("thigh_joint_fr", "leg_joint_fr"),
            ("thigh_joint_rl", "leg_joint_rl"),
            ("thigh_joint_rr", "leg_joint_rr"),
        ]

        self.joint_ids: List[Tuple[int, int]] = []
        self.actuator_ids: List[Tuple[int, int]] = []
        self._cache_joint_and_actuator_ids()
        self._cache_joint_qpos_dof_adrs()
        self._actuator_idx_leg = np.array(
            [int(a) for pair in self.actuator_ids for a in pair if int(a) >= 0],
            dtype=np.int32,
        )
        self._dof_idx_leg = np.array(
            [int(self.model.jnt_dofadr[int(self.model.actuator_trnid[act_id, 0])]) for act_id in self._actuator_idx_leg],
            dtype=np.int32,
        )

        # Energy tank (optional): joint-space virtual impedance + tank + admittance -> q_ref
        # 参数来自 compute_energy_tank_params.py：K_j/D_j/A 按模型 J_eff 计算，E0/E_max 按 rollout 实测建议
        self._energy_tank: Optional[EnergyTankJointSpace] = None
        if self.use_energy_tank:
            K0, D0 = 0.102893, 0.011463
            if self.et_vector_kd:
                kh = float(et_k_hip) if et_k_hip is not None else 1.2 * K0
                kk = float(et_k_knee) if et_k_knee is not None else 0.8 * K0
                dh = float(et_d_hip) if et_d_hip is not None else 1.2 * D0
                dk = float(et_d_knee) if et_d_knee is not None else 0.8 * D0
                K_arr = np.array([kh, kk] * 4, dtype=np.float64)
                D_arr = np.array([dh, dk] * 4, dtype=np.float64)
            else:
                K_arr, D_arr = K0, D0
            self._energy_tank = EnergyTankJointSpace(
                n_joints=8,
                dt=self.dt,
                E0=self.et_tank_E0,
                E_max=self.et_tank_E_max,
                K_j=K_arr,
                D_j=D_arr,
                A=self.et_tank_A,
                eta=self.et_tank_eta,
            )
            self._et_k_base_8 = np.asarray(K_arr if isinstance(K_arr, np.ndarray) else [float(K_arr)] * 8, dtype=np.float64)
            self._et_d_base_8 = np.asarray(D_arr if isinstance(D_arr, np.ndarray) else [float(D_arr)] * 8, dtype=np.float64)

        # Episode state
        self.sim_step_counter = 0
        self.control_step_counter = 0
        self._ep_energy = 0.0
        self._ep_distance = 0.0
        self._last_base_y = 0.0
        self._vx0 = 0.0
        self.tau_max = 0.157

        # last-step tau cache
        self._last_tau_8 = np.zeros(8, dtype=np.float64)

        # Energy tank: previous IK setpoint (same order as q_d_8) for causal qd_d ≈ Δq_d/dt
        self._et_prev_q_d_8: Optional[np.ndarray] = None

        # Scheme B: cache last valid joint targets per leg
        self._last_valid_joint_targets: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(4)]

        # action-state (8D)：初始为 ACTION_RANGES 中点，与训练范围 (0.3,1) Hz、(0.2,1.0) mu 一致
        self._cur_f4 = np.ones(4, dtype=np.float64) * float(np.mean(self.ACTION_RANGES["f"]))
        self._cur_mu4 = np.ones(4, dtype=np.float64) * float(np.mean(self.ACTION_RANGES["mu"]))
        # 做法B: 实际本步使用的动作；仅当 control_step % action_update_interval == 0 时刷新
        self._last_applied_action = None

        # Episode accumulators
        self._reset_episode_accumulators()

        # per-step CSV log config
        self.enable_csv_log = bool(enable_csv_log)
        self.csv_path = str(csv_path)
        self.auto_timestamp_csv = bool(auto_timestamp_csv)
        self.print_sim_step_torque = bool(print_sim_step_torque)
        self.print_sim_step_joint_angles = bool(print_sim_step_joint_angles)
        self._csv_fp = None
        self._csv_writer = None
        self._csv_header_written = False
        self._contact_E_friction_step_acc = 0.0
        self._contact_E_normal_step_acc = 0.0
        self._contact_E_step_acc = 0.0

        # episode summary config
        self.enable_episode_summary = bool(enable_episode_summary)
        self.episode_summary_path = str(episode_summary_path)
        self.auto_timestamp_episode_summary = bool(auto_timestamp_episode_summary)
        self._ep_summary_path_resolved = None

        # stance/swing abs work accumulators (8 actuators)
        self._ep_stance_abs_work_8 = np.zeros(8, dtype=np.float64)
        self._ep_swing_abs_work_8 = np.zeros(8, dtype=np.float64)

        # Optional viewer for eval: when set, step() syncs and sleeps each sim step for smooth real-time display
        self._viewer = None

        # Optional callback(sim_step, joint_targets, actual_qpos_8) called after each mj_step (for eval per-sim-step print)
        self.sim_step_callback = None

        # Gym spaces: action is 8D: [f4, mu4]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # obs_dim: (from sensors) base_height 1 + lin_vel 3 + ang_vel 3 + joint_pos 8 + contacts 4 + phase_sin_cos 8
        #          = 27, plus controller-internal cur_f4 4 + cur_mu4 4 => 35; +2 (e_tank_norm, alpha_tank) when use_energy_tank
        obs_dim = 37 if self.use_energy_tank else 35
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Subclasses (e.g. direct-joint RL) may set this to a length-4 list of (knee, hip) targets
        # in the same format as IK output, bypassing CPG+IK for that macro-step.
        self._direct_joint_targets_override: Optional[List[Tuple[float, float]]] = None

    @staticmethod
    def _resolve_gait_template_offsets(gait_template: str) -> np.ndarray:
        """
        Return fixed phase offsets [FL, FR, RL, RR] for a gait template.
        """
        g = str(gait_template).strip().lower()
        if g == "trot":
            return np.array([0.0, math.pi, math.pi, 0.0], dtype=float)
        if g == "pace":
            return np.array([0.0, math.pi, 0.0, math.pi], dtype=float)
        if g == "bound":
            return np.array([0.0, 0.0, math.pi, math.pi], dtype=float)
        if g == "walk":
            # Simple four-beat template; may require leg-order tuning in practice.
            return np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=float)
        raise ValueError(
            f"Unsupported gait_template='{gait_template}'. "
            "Supported: trot, pace, bound, walk."
        )

    def _et_group_scale_vec_8(self) -> np.ndarray:
        """
        Trot diagonal groups: A = legs {0,3} (FL, RR), B = {1,2} (FR, RL).
        Scale repeats per joint in hip/knee order [FL, FR, RL, RR] -> indices 0:2 and 6:8 use w_A, 2:6 use w_B.
        """
        p = [float(self.foot_path.get_leg_phase(i)) for i in range(4)]
        phi_A = math.atan2(math.sin(p[0]) + math.sin(p[3]), math.cos(p[0]) + math.cos(p[3]))
        phi_B = math.atan2(math.sin(p[1]) + math.sin(p[2]), math.cos(p[1]) + math.cos(p[2]))
        w_lo, w_hi = self.et_w_mod_min, self.et_w_mod_max
        w_A = w_lo + (w_hi - w_lo) * (0.5 + 0.5 * math.sin(phi_A))
        w_B = w_lo + (w_hi - w_lo) * (0.5 + 0.5 * math.sin(phi_B))
        s = np.ones(8, dtype=np.float64)
        s[0:2] = w_A
        s[2:6] = w_B
        s[6:8] = w_A
        return s

    def _et_coupling_mats_8(self, scale_8: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build explicit joint-coupling matrices for C:
          - within-leg hip-knee 2x2 coupling (always when et_joint_coupling=True)
          - optional diagonal-leg same-joint coupling (FL<->RR, FR<->RL)
        """
        CK = np.zeros((8, 8), dtype=np.float64)
        CD = np.zeros((8, 8), dtype=np.float64)
        s = np.ones(8, dtype=np.float64) if scale_8 is None else np.asarray(scale_8, dtype=np.float64).ravel()
        if s.size != 8:
            s = np.ones(8, dtype=np.float64)
        k_eff = self._et_k_base_8 * s
        d_eff = self._et_d_base_8 * s

        # within-leg hip<->knee coupling
        for leg in range(4):
            h = 2 * leg
            k = h + 1
            ck = self.et_couple_k_ratio * float(np.sqrt(max(k_eff[h] * k_eff[k], 0.0)))
            cd = self.et_couple_d_ratio * float(np.sqrt(max(d_eff[h] * d_eff[k], 0.0)))
            CK[h, k] = CK[k, h] = ck
            CD[h, k] = CD[k, h] = cd

        # optional diagonal-leg same-joint coupling
        if self.et_diag_leg_coupling:
            diag_pairs = [(0, 6), (2, 4), (1, 7), (3, 5)]  # hip A/B then knee A/B
            for i, j in diag_pairs:
                ck = self.et_diag_leg_k_ratio * float(np.sqrt(max(k_eff[i] * k_eff[j], 0.0)))
                cd = self.et_diag_leg_d_ratio * float(np.sqrt(max(d_eff[i] * d_eff[j], 0.0)))
                CK[i, j] = CK[j, i] = ck
                CD[i, j] = CD[j, i] = cd
        return CK, CD

    def get_obs_layout(self) -> List[Tuple[str, str]]:
        """Return list of (obs_name, index_or_slice) for each observation component (for logging)."""
        layout: List[Tuple[str, str]] = [
            ("base_height", "0"),
            ("lin_vel", "1:4"),
            ("ang_vel", "4:7"),
            ("joint_pos", "7:15"),
            ("contact_flags", "15:19"),
            ("phase_sin_cos", "19:27"),
            ("cur_f4", "27:31"),
            ("cur_mu4", "31:35"),
        ]
        if self.use_energy_tank:
            layout.append(("e_tank_norm", "35"))
            layout.append(("alpha_tank", "36"))
        return layout

    def set_viewer(self, viewer) -> None:
        """Set viewer for per-sim-step sync + sleep in step() (smooth real-time eval)."""
        self._viewer = viewer

    def set_sim_step_callback(self, callback) -> None:
        """Set callback(sim_step, joint_targets, actual_qpos_8) called after each mj_step. None to disable."""
        self.sim_step_callback = callback

    # ------------------ episode accumulators ------------------

    def _reset_episode_accumulators(self) -> None:
        self._ep_total_reward = 0.0
        self._ep_comp_sums: Dict[str, float] = {
            "r_vel": 0.0,
            "r_pose": 0.0,
            "r_xvel": 0.0,
            "r_sat": 0.0,
            "r_joint_energy": 0.0,
            "r_impact": 0.0,
            "r_body": 0.0,
            "r_smooth": 0.0,
        }
        self._ep_abs_power_sum = 0.0
        self._ep_abs_work = 0.0
        self._ep_step_count = 0
        self._prev_contacts_step = [0.0, 0.0, 0.0, 0.0]

        self._ep_stance_abs_work_8 = np.zeros(8, dtype=np.float64)
        self._ep_swing_abs_work_8 = np.zeros(8, dtype=np.float64)

    # ------------------ ID mapping ------------------

    def _cache_joint_and_actuator_ids(self) -> None:
        self.joint_ids.clear()
        self.actuator_ids.clear()
        for hip_name, knee_name in self.controlled_joint_names:
            hip_j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, hip_name)
            knee_j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, knee_name)
            self.joint_ids.append((hip_j, knee_j))

            hip_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, hip_name)
            knee_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, knee_name)
            self.actuator_ids.append((hip_a, knee_a))

    def _cache_joint_qpos_dof_adrs(self) -> None:
        """Cache qpos/dof indices for 8 controlled joints (hip, knee per leg) for energy tank."""
        self._joint_qpos_adrs: List[int] = []
        self._joint_dof_adrs: List[int] = []
        for hip_id, knee_id in self.joint_ids:
            if hip_id >= 0:
                self._joint_qpos_adrs.append(int(self.model.jnt_qposadr[hip_id]))
                self._joint_dof_adrs.append(int(self.model.jnt_dofadr[hip_id]))
            else:
                self._joint_qpos_adrs.append(-1)
                self._joint_dof_adrs.append(-1)
            if knee_id >= 0:
                self._joint_qpos_adrs.append(int(self.model.jnt_qposadr[knee_id]))
                self._joint_dof_adrs.append(int(self.model.jnt_dofadr[knee_id]))
            else:
                self._joint_qpos_adrs.append(-1)
                self._joint_dof_adrs.append(-1)

    # ------------------ per-step CSV helpers ------------------

    def _close_csv(self) -> None:
        if self._csv_fp is not None:
            try:
                self._csv_fp.flush()
                self._csv_fp.close()
            except Exception:
                pass
        self._csv_fp = None
        self._csv_writer = None
        self._csv_header_written = False

    def _resolve_csv_path_for_this_episode(self) -> str:
        p = self.csv_path
        if p.endswith("/") or (os.path.exists(p) and os.path.isdir(p)):
            os.makedirs(p, exist_ok=True)
            base = "rat_debug"
            suffix = ".csv"
            if self.auto_timestamp_csv:
                ts = time.strftime("%Y%m%d_%H%M%S")
                return os.path.join(p, f"{base}_{ts}{suffix}")
            return os.path.join(p, f"{base}{suffix}")

        if self.auto_timestamp_csv:
            root, ext = os.path.splitext(p)
            if ext.lower() != ".csv":
                ext = ".csv"
            ts = time.strftime("%Y%m%d_%H%M%S")
            return f"{root}_{ts}{ext}"
        return p

    def _open_csv_for_episode(self) -> None:
        if not self.enable_csv_log:
            return
        self._close_csv()

        out_path = self._resolve_csv_path_for_this_episode()
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._csv_fp = open(out_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fp)
        self._csv_header_written = False

    def _write_csv_header_if_needed(self) -> None:
        if not self.enable_csv_log:
            return
        if self._csv_writer is None or self._csv_header_written:
            return

        header: List[str] = [
            "sim_step",
            "dt",
            "abs_power_W",
            "ep_index_energy_J",
            "base_vy_qvel",
            "roll",
            "pitch",
            "base_vy_abs",
        ]
        if self.enable_extended_logging:
            header += [
                "base_h",
                "base_vx",
                "base_vy",
                "base_vz",
                "base_wx",
                "base_wy",
                "base_wz",
                "contact_E_step",
                "contact_E_friction_step",
                "contact_E_normal_step",
            ]

        header += ["contact_fl", "contact_fr", "contact_rl", "contact_rr"]
        header += ["phi_fl", "phi_fr", "phi_rl", "phi_rr"]

        for i in range(4):
            header.append(f"absP_leg{i}_hip_W")
            header.append(f"absP_leg{i}_knee_W")
        for i in range(4):
            header.append(f"Estep_leg{i}_hip_J")
            header.append(f"Estep_leg{i}_knee_J")

        for i in range(4):
            header.append(f"tau_leg{i}_hip")
            header.append(f"tau_leg{i}_knee")

        for i in range(4):
            header.append(f"qdes_leg{i}_hip")
            header.append(f"qdes_leg{i}_knee")

        for i in range(4):
            header.append(f"qvel_leg{i}_hip")
            header.append(f"qvel_leg{i}_knee")
            header.append(f"qerr_leg{i}_hip")
            header.append(f"qerr_leg{i}_knee")

        self._csv_writer.writerow(header)
        self._csv_header_written = True

    # ------------------ contacts (相位 or SENSORDATA 12-15) ------------------

    def _get_contacts_step(self) -> List[float]:
        """
        use_phase_for_contact=True：与 sim_test 一致，φ≥π 为 stance（在后方往前推的那半周）
        use_phase_for_contact=False：用 MuJoCo sensordata 12-15 物理接触
        """
        if self.use_phase_for_contact:
            return [
                1.0 if self.foot_path.get_leg_phase(i) >= math.pi else 0.0
                for i in range(4)
            ]
        try:
            sd = np.asarray(self.data.sensordata, dtype=np.float64).reshape(-1)
            if sd.size >= 16:
                c = sd[self.SENSOR_TOUCH_SLICE]
                return [1.0 if float(x) > 0.0 else 0.0 for x in c.tolist()]
        except Exception:
            pass
        return [0.0, 0.0, 0.0, 0.0]

    # ------------------ per-step CSV ------------------

    def _log_step_to_csv(
        self,
        *,
        abs_power: float,
        ep_index_energy: float,
        joint_targets: List[Tuple[float, float]],
        contacts: List[float],
        joint_absP_8: np.ndarray,
        joint_Estep_8: np.ndarray,
        roll: float = 0.0,
        pitch: float = 0.0,
        sim_step_override: Optional[int] = None,
    ) -> None:
        if not self.enable_csv_log:
            return
        if self._csv_writer is None:
            return
        self._write_csv_header_if_needed()
        base_vy = float(self.data.qvel[1])
        base_vy_abs = float(abs(base_vy))
        step_for_row = int(sim_step_override) if sim_step_override is not None else int(self.sim_step_counter)

        row: List[float] = [
            float(step_for_row),
            float(self.dt),
            float(abs_power),
            float(ep_index_energy),
            float(base_vy),
            float(roll),
            float(pitch),
            float(base_vy_abs),
        ]
        if self.enable_extended_logging:
            row += [
                float(self.data.qpos[2]),
                float(self.data.qvel[0]),
                float(self.data.qvel[1]),
                float(self.data.qvel[2]),
                float(self.data.qvel[3]),
                float(self.data.qvel[4]),
                float(self.data.qvel[5]),
                float(self._contact_E_step_acc),
                float(self._contact_E_friction_step_acc),
                float(self._contact_E_normal_step_acc),
            ]

        c = (contacts + [0.0, 0.0, 0.0, 0.0])[:4]
        row += [float(c[0]), float(c[1]), float(c[2]), float(c[3])]
        row += [float(self.foot_path.get_leg_phase(i)) for i in range(4)]

        jp = np.asarray(joint_absP_8, dtype=np.float64).reshape(-1)
        if jp.size < 8:
            jp = np.pad(jp, (0, 8 - jp.size))
        row += [float(x) for x in jp[:8]]

        je = np.asarray(joint_Estep_8, dtype=np.float64).reshape(-1)
        if je.size < 8:
            je = np.pad(je, (0, 8 - je.size))
        row += [float(x) for x in je[:8]]

        for (hip_act, knee_act) in self.actuator_ids:
            tau_h = float(self.data.actuator_force[int(hip_act)]) if hip_act >= 0 else 0.0
            tau_k = float(self.data.actuator_force[int(knee_act)]) if knee_act >= 0 else 0.0
            row.append(tau_h)
            row.append(tau_k)

        for (hip_id, knee_id), (q1_des, q2_des) in zip(self.joint_ids, joint_targets):
            row.append(float(q1_des))
            row.append(float(q2_des))
            if hip_id >= 0:
                qpos_adr = int(self.model.jnt_qposadr[hip_id])
                dof_adr = int(self.model.jnt_dofadr[hip_id])
                q = float(self.data.qpos[qpos_adr])
                qd = float(self.data.qvel[dof_adr])
                qerr = float(q1_des - q)
            else:
                qd = 0.0
                qerr = 0.0

            if knee_id >= 0:
                qpos_adr = int(self.model.jnt_qposadr[knee_id])
                dof_adr = int(self.model.jnt_dofadr[knee_id])
                q2 = float(self.data.qpos[qpos_adr])
                qd2 = float(self.data.qvel[dof_adr])
                qerr2 = float(q2_des - q2)
            else:
                qd2 = 0.0
                qerr2 = 0.0

            row.append(qd)
            row.append(qd2)
            row.append(qerr)
            row.append(qerr2)

        self._csv_writer.writerow(row)

    # ------------------ episode summary helpers ------------------

    def _resolve_episode_summary_path(self) -> str:
        p = self.episode_summary_path

        if p.endswith("/") or (os.path.exists(p) and os.path.isdir(p)):
            os.makedirs(p, exist_ok=True)
            base = "rat_episode_summary"
            suffix = ".csv"
            if self.auto_timestamp_episode_summary:
                ts = time.strftime("%Y%m%d_%H%M%S")
                return os.path.join(p, f"{base}_{ts}{suffix}")
            return os.path.join(p, f"{base}{suffix}")

        if self.auto_timestamp_episode_summary:
            root, ext = os.path.splitext(p)
            if ext.lower() != ".csv":
                ext = ".csv"
            ts = time.strftime("%Y%m%d_%H%M%S")
            return f"{root}_{ts}{ext}"
        return p

    def _episode_summary_header(self) -> List[str]:
        cols = [
            "timestamp",
            "episode_steps",
            "forward_dist_negY_m",
            "stance_abs_work_total_J",
            "swing_abs_work_total_J",
            "front_thigh_stance_J",
            "front_thigh_swing_J",
            "front_calf_stance_J",
            "front_calf_swing_J",
            "rear_thigh_stance_J",
            "rear_thigh_swing_J",
            "rear_calf_stance_J",
            "rear_calf_swing_J",
        ]
        for i in range(4):
            cols.append(f"leg{i}_hip_stance_J")
            cols.append(f"leg{i}_hip_swing_J")
            cols.append(f"leg{i}_knee_stance_J")
            cols.append(f"leg{i}_knee_swing_J")
        return cols

    def _append_episode_summary_row(self) -> None:
        if not self.enable_episode_summary:
            return

        out_path = self._ep_summary_path_resolved or self._resolve_episode_summary_path()
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        file_exists = os.path.exists(out_path)
        need_header = (not file_exists) or (os.path.getsize(out_path) == 0)

        stance8 = self._ep_stance_abs_work_8.copy()
        swing8 = self._ep_swing_abs_work_8.copy()

        stance_total = float(np.sum(stance8))
        swing_total = float(np.sum(swing8))

        front_thigh_st = float(stance8[0] + stance8[2])
        front_thigh_sw = float(swing8[0] + swing8[2])
        front_calf_st  = float(stance8[1] + stance8[3])
        front_calf_sw  = float(swing8[1] + swing8[3])

        rear_thigh_st = float(stance8[4] + stance8[6])
        rear_thigh_sw = float(swing8[4] + swing8[6])
        rear_calf_st  = float(stance8[5] + stance8[7])
        rear_calf_sw  = float(swing8[5] + swing8[7])

        ts = time.strftime("%Y%m%d_%H%M%S")

        row: List[Any] = [
            ts,
            int(self._ep_step_count),
            float(self._ep_distance),
            stance_total,
            swing_total,
            front_thigh_st, front_thigh_sw,
            front_calf_st,  front_calf_sw,
            rear_thigh_st,  rear_thigh_sw,
            rear_calf_st,   rear_calf_sw,
        ]

        for leg in range(4):
            hip_idx = leg * 2 + 0
            knee_idx = leg * 2 + 1
            row += [
                float(stance8[hip_idx]), float(swing8[hip_idx]),
                float(stance8[knee_idx]), float(swing8[knee_idx]),
            ]

        with open(out_path, "a", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            if need_header:
                w.writerow(self._episode_summary_header())
            w.writerow(row)

    # ------------------ energy ------------------

    def _abs_power_step(self) -> float:
        if self.model.nu <= 0:
            return 0.0
        p = 0.0
        for (hip_act, knee_act) in self.actuator_ids:
            if hip_act >= 0:
                act = int(hip_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    tau = float(self.data.actuator_force[act])
                    qd = float(self.data.qvel[dof])
                    p += abs(tau * qd)
            if knee_act >= 0:
                act = int(knee_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    tau = float(self.data.actuator_force[act])
                    qd = float(self.data.qvel[dof])
                    p += abs(tau * qd)
        return float(p)

    def _joint_abs_power_step(self) -> np.ndarray:
        out = np.zeros(8, dtype=np.float64)
        if self.model.nu <= 0:
            return out

        idx = 0
        for (hip_act, knee_act) in self.actuator_ids:
            if hip_act >= 0:
                act = int(hip_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    tau = float(self.data.actuator_force[act])
                    qd = float(self.data.qvel[dof])
                    out[idx] = abs(tau * qd)
            idx += 1

            if knee_act >= 0:
                act = int(knee_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    tau = float(self.data.actuator_force[act])
                    qd = float(self.data.qvel[dof])
                    out[idx] = abs(tau * qd)
            idx += 1

        return out

    def _joint_abs_work_step_8(self) -> np.ndarray:
        return self._joint_abs_power_step() * float(self.dt)

    def _joint_power_split_step(self) -> Tuple[np.ndarray, np.ndarray]:
        p_pos = np.zeros(8, dtype=np.float64)
        p_neg = np.zeros(8, dtype=np.float64)
        if self.model.nu <= 0:
            return p_pos, p_neg
        idx = 0
        for (hip_act, knee_act) in self.actuator_ids:
            if hip_act >= 0:
                act = int(hip_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    p = float(self.data.actuator_force[act]) * float(self.data.qvel[dof])
                    p_pos[idx] = max(p, 0.0)
                    p_neg[idx] = max(-p, 0.0)
            idx += 1
            if knee_act >= 0:
                act = int(knee_act)
                j_id = int(self.model.actuator_trnid[act, 0])
                if j_id >= 0:
                    dof = int(self.model.jnt_dofadr[j_id])
                    p = float(self.data.actuator_force[act]) * float(self.data.qvel[dof])
                    p_pos[idx] = max(p, 0.0)
                    p_neg[idx] = max(-p, 0.0)
            idx += 1
        return p_pos, p_neg

    @staticmethod
    def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float]:
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        return roll, pitch

    # ------------------ action mapping (8D) ------------------

    def _map_action_8(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        action shape (8,) in [-1,1]:
          [ f_FL, f_FR, f_RL, f_RR,  mu_FL, mu_FR, mu_RL, mu_RR ]
        mapped to physical ranges as in your scan script.
        """
        a = np.clip(action.astype(np.float64), -1.0, 1.0).reshape(8)

        f_lo, f_hi = self.ACTION_RANGES["f"]
        m_lo, m_hi = self.ACTION_RANGES["mu"]

        f4 = f_lo + (a[0:4] + 1.0) * 0.5 * (f_hi - f_lo)
        mu4 = m_lo + (a[4:8] + 1.0) * 0.5 * (m_hi - m_lo)
        return f4, mu4

    def _smooth_vec(self, cur: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        return (1.0 - self.smoothing_alpha) * cur + self.smoothing_alpha * tgt

    # ------------------ Gym API ------------------

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.sim_step_counter = 0
        self.control_step_counter = 0
        self._ep_energy = 0.0
        self._ep_distance = 0.0

        self._reset_episode_accumulators()

        if self.enable_csv_log:
            self._open_csv_for_episode()

        if self.enable_episode_summary:
            self._ep_summary_path_resolved = self._resolve_episode_summary_path()

        # baseline reset
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._et_prev_q_d_8 = None
        if self.use_energy_tank and self._energy_tank is not None:
            q_8 = np.array(
                [float(self.data.qpos[adr]) if adr >= 0 else 0.0 for adr in self._joint_qpos_adrs],
                dtype=np.float64,
            )
            if q_8.size == 8:
                self._energy_tank.reset(q_8)

        self._last_base_y = float(self.data.qpos[1])
        self._vx0 = float(self.data.qvel[0])
        self._last_tau_8[:] = 0.0

        self._last_valid_joint_targets = [(0.0, 0.0) for _ in range(4)]

        # init current controls (midpoints) ✅
        f_mid = float(np.mean(self.ACTION_RANGES["f"]))
        mu_mid = float(np.mean(self.ACTION_RANGES["mu"]))

        self._cur_f4[:] = f_mid
        self._cur_mu4[:] = mu_mid
        if self.init_f4_hz is not None:
            f_lo, f_hi = self.ACTION_RANGES["f"]
            self._cur_f4[:] = float(np.clip(self.init_f4_hz, f_lo, f_hi))

        if self.randomize_cpg_params:
            f_lo, f_hi = self.ACTION_RANGES["f"]
            m_lo, m_hi = self.ACTION_RANGES["mu"]

            self._cur_f4[:] = np.clip(
                self._cur_f4 + self.np_random.normal(0.0, self.reset_f_sigma, size=4),
                f_lo, f_hi
            )

            self._cur_mu4[:] = np.clip(
                self._cur_mu4 + self.np_random.normal(0.0, self.reset_mu_sigma, size=4),
                m_lo, m_hi
            )

        if self.fixed_cpg:
            self._cur_f4[:] = float(getattr(self, "_fixed_f_override", 0.5))
            self._cur_mu4[:] = float(getattr(self, "_fixed_mu_override", 1.0))
            # 打印 actuator 映射与 ctrl 顺序，便于与 sim_test 对比
            leg_names = ["FL", "FR", "RL", "RR"]
            print("[RL fixed_cpg] actuator_ids (hip_act, knee_act) per leg:", [(leg_names[i], self.actuator_ids[i]) for i in range(4)])
            nu = min(8, self.model.nu)
            print("[RL fixed_cpg] model actuator order [0..%d]:" % (nu - 1), end="")
            for idx in range(nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
                print(" %d=%s" % (idx, name or "?"), end="")
            print()

        desired = self.base_desired_offsets  # FIXED selected gait template

        # Randomize global phase delta while keeping desired layout
        if self.randomize_cpg_phase:
            delta = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            phases = (desired + delta) % (2.0 * math.pi)
            self.foot_path.set_controls(self._cur_f4, self._cur_mu4)
            self.foot_path.reset(phases=phases.tolist())
        else:
            self.foot_path.set_controls(self._cur_f4, self._cur_mu4)
            self.foot_path.reset(phases=desired.tolist())

        # 做法B: 新 episode 尚未执行任何 step，实际动作为空
        self._last_applied_action = None

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "ep_distance": 0.0,
            "cot": 0.0,
            "ik_fail_legs": 0,
            "ik_fail_rate": 0.0,
        }
        return obs, info

    def _get_obs(self) -> np.ndarray:
        # Prefer IMU-based sensors for base pose/velocity; fall back to qpos/qvel if unavailable.
        sd = np.asarray(self.data.sensordata, dtype=np.float64).reshape(-1)

        if sd.size >= self.IMU_POS_SLICE.stop:
            com_pos = sd[self.IMU_POS_SLICE]
            base_height = float(com_pos[2])
        else:
            base_height = float(self.data.qpos[2])

        if sd.size >= self.IMU_VEL_SLICE.stop:
            lin_vel = sd[self.IMU_VEL_SLICE].astype(np.float64)
        else:
            lin_vel = self.data.qvel[0:3].astype(np.float64)

        if sd.size >= self.IMU_GYRO_SLICE.stop:
            ang_vel = sd[self.IMU_GYRO_SLICE].astype(np.float64)
        else:
            ang_vel = self.data.qvel[3:6].astype(np.float64)

        # Joint positions from servo angle sensors (0–7); if unavailable fall back to qpos.
        joint_pos: List[float] = []
        if sd.size >= 8:
            joint_pos.extend([float(x) for x in sd[0:8]])
        else:
            for hip_id, knee_id in self.joint_ids:
                if hip_id >= 0:
                    qpos_adr = int(self.model.jnt_qposadr[hip_id])
                    joint_pos.append(float(self.data.qpos[qpos_adr]))
                else:
                    joint_pos.append(0.0)

                if knee_id >= 0:
                    qpos_adr = int(self.model.jnt_qposadr[knee_id])
                    joint_pos.append(float(self.data.qpos[qpos_adr]))
                else:
                    joint_pos.append(0.0)

        # contacts from sensors 12-15
        contact_flags = self._get_contacts_step()

        phase_sin_cos: List[float] = []
        for i in range(4):
            phi = self.foot_path.get_leg_phase(i)
            phase_sin_cos.append(math.sin(phi))
            phase_sin_cos.append(math.cos(phi))

        obs_parts: List[np.ndarray] = [
            np.array([base_height], dtype=np.float64),
            lin_vel,
            ang_vel,
            np.array(joint_pos, dtype=np.float64),
            np.array(contact_flags, dtype=np.float64),
            np.array(phase_sin_cos, dtype=np.float64),
            self._cur_f4.astype(np.float64),
            self._cur_mu4.astype(np.float64),
        ]
        if self.use_energy_tank and self._energy_tank is not None:
            e_max = max(float(self._energy_tank.E_max), 1e-9)
            e_tank_norm = float(self._energy_tank.E_t) / e_max
            alpha_tank = float(self._energy_tank.alpha_last)
            obs_parts.append(np.array([e_tank_norm, alpha_tank], dtype=np.float64))
        obs = np.concatenate(obs_parts, axis=0)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action.shape != (8,):
            raise ValueError("Action must be an 8-dimensional vector: [f4, mu4]")

        # CPG 参数 (f, mu)：做法B — 仅每 action_update_interval 个 env step 更新一次，中间步沿用上一拍的 (f,μ)
        # fixed_cpg=True 时忽略 action，始终用 f4=0.5、mu4=1.0（与 sim_test 一致）
        if self.control_step_counter % self.action_update_interval == 0:
            if self.fixed_cpg:
                f_fixed = float(getattr(self, "_fixed_f_override", 0.5))
                mu_fixed = float(getattr(self, "_fixed_mu_override", 1.0))
                self._cur_f4 = np.full(4, f_fixed, dtype=np.float64)
                self._cur_mu4 = np.full(4, mu_fixed, dtype=np.float64)
                self.foot_path.set_controls(self._cur_f4, self._cur_mu4)
                self._last_applied_action = np.array(action, dtype=np.float64)  # 仅用于 info/日志
            else:
                f4_tgt, mu4_tgt = self._map_action_8(action)
                self._cur_f4 = self._smooth_vec(self._cur_f4, f4_tgt)
                self._cur_mu4 = self._smooth_vec(self._cur_mu4, mu4_tgt)
                self.foot_path.set_controls(self._cur_f4, self._cur_mu4)
                self._last_applied_action = np.array(action, dtype=np.float64)

        K = self.n_substeps
        # Optional IDER (Inverse-Dynamics-Informed Excessive-effort Regularization): reset per-macro-step accumulators
        if hasattr(self, "_ider_begin_macro_step"):
            self._ider_begin_macro_step()
        acc_abs_power = 0.0
        acc_joint_absP_8 = np.zeros(8, dtype=np.float64)
        acc_joint_posP_8 = np.zeros(8, dtype=np.float64)
        acc_joint_negP_8 = np.zeros(8, dtype=np.float64)
        impact_sum = 0.0
        impact_count = 0
        acc_fwd_vel = 0.0
        acc_roll = 0.0
        acc_pitch = 0.0
        acc_base_vx = 0.0
        acc_base_h = 0.0
        acc_base_vz = 0.0
        self._contact_E_friction_step_acc = 0.0
        self._contact_E_normal_step_acc = 0.0
        self._contact_E_step_acc = 0.0
        step_base_y_start = float(self.data.qpos[1])
        base_sim = self.sim_step_counter
        last_E_tank = 0.0
        last_alpha_tank = 1.0

        for i in range(K):
            # CPG 相位：每个 sim step 更新一次
            self.foot_path.step()

            # 1–3) CPG -> foot -> IK, or direct joint targets (no-CPG ablation hook)
            if self._direct_joint_targets_override is not None:
                foot_targets = [(0.0, 0.0)] * 4
                joint_targets = self._direct_joint_targets_override
            else:
                # 1) CPG -> foot targets
                foot_targets = []
                for leg_id in range(4):
                    Fy, Fz = self.foot_path.get_foot_target(leg_id=leg_id)
                    foot_targets.append((Fy, Fz))

                # 2) turn + 3) IK -> joint targets
                joint_targets = []
                for leg_id, (Fy, Fz) in enumerate(foot_targets):
                    turn = self.turn_F if leg_id < 2 else self.turn_H
                    tY = math.cos(turn) * Fy - math.sin(turn) * Fz
                    tZ = math.cos(turn) * Fz + math.sin(turn) * Fy
                    q = self.leg_models[leg_id].pos_2_angle(tY, tZ)
                    if q is None:
                        q1, q2 = self._last_valid_joint_targets[leg_id]
                    else:
                        q1, q2 = q
                        self._last_valid_joint_targets[leg_id] = (q1, q2)
                    joint_targets.append((float(q1), float(q2)))

            # 4) Apply to actuators（可选：经 energy tank 得到 q_ref 再写 ctrl）
            if self.model.nu > 0:
                ctrl = np.zeros(self.model.nu, dtype=np.float64)
                if self.use_energy_tank and self._energy_tank is not None:
                    # q_d_8 与 _joint_qpos_adrs 同序：hip, knee 即 (q2, q1) 每条腿
                    q_d_8 = np.array(
                        [
                            joint_targets[0][1], joint_targets[0][0],
                            joint_targets[1][1], joint_targets[1][0],
                            joint_targets[2][1], joint_targets[2][0],
                            joint_targets[3][1], joint_targets[3][0],
                        ],
                        dtype=np.float64,
                    )
                    q_8 = np.array(
                        [float(self.data.qpos[adr]) if adr >= 0 else 0.0 for adr in self._joint_qpos_adrs],
                        dtype=np.float64,
                    )
                    qd_8 = np.array(
                        [float(self.data.qvel[adr]) if adr >= 0 else 0.0 for adr in self._joint_dof_adrs],
                        dtype=np.float64,
                    )
                    et_dt = float(self.dt)
                    if self._et_prev_q_d_8 is None:
                        qd_d_8 = np.zeros(8, dtype=np.float64)
                    else:
                        qd_d_8 = (q_d_8 - self._et_prev_q_d_8) / et_dt
                    scale_8 = self._et_group_scale_vec_8() if self.et_group_scale else None
                    if self.et_joint_coupling:
                        couple_scale = scale_8 if self.et_couple_phase_gate else None
                        CK, CD = self._et_coupling_mats_8(couple_scale)
                    else:
                        CK, CD = None, None
                    q_ref_8, last_E_tank, last_alpha_tank = self._energy_tank.step(
                        q_d_8,
                        q_8,
                        qd_8,
                        scale=scale_8,
                        coupling_K=CK,
                        coupling_D=CD,
                        qd_d=qd_d_8,
                    )
                    self._et_prev_q_d_8 = q_d_8.copy()
                    # q_ref_8 顺序 [hip_fl, knee_fl, ...]，ctrl: knee_act←q_ref[2*i+1], hip_act←q_ref[2*i]
                    for leg_i, (hip_act, knee_act) in enumerate(self.actuator_ids):
                        if knee_act >= 0:
                            ctrl[int(knee_act)] = q_ref_8[2 * leg_i + 1]
                        if hip_act >= 0:
                            ctrl[int(hip_act)] = q_ref_8[2 * leg_i]
                else:
                    for (hip_act, knee_act), (q1_des, q2_des) in zip(self.actuator_ids, joint_targets):
                        if knee_act >= 0:
                            ctrl[int(knee_act)] = q1_des
                        if hip_act >= 0:
                            ctrl[int(hip_act)] = q2_des
                self.data.ctrl[:] = ctrl
                # fixed_cpg 时第一步打印 joint_targets 与 ctrl[:8]，便于与 sim_test 对比
                if self.fixed_cpg and self.sim_step_counter == 0 and i == 0:
                    flat_jt = [joint_targets[leg][j] for leg in range(4) for j in range(2)]
                    print("[RL fixed_cpg] step0 joint_targets (FL_hip, FL_knee, FR_hip, FR_knee, RL_hip, RL_knee, RR_hip, RR_knee) =", [round(x, 5) for x in flat_jt])
                    print("[RL fixed_cpg] step0 ctrl[:8] =", [round(float(x), 5) for x in ctrl[:8]])

            mujoco.mj_step(self.model, self.data)
            if self.enable_extended_logging:
                for ic in range(int(self.data.ncon)):
                    contact = self.data.contact[ic]
                    g1, g2 = int(contact.geom[0]), int(contact.geom[1])
                    b1 = int(self.model.geom_bodyid[g1])
                    b2 = int(self.model.geom_bodyid[g2])
                    # Keep only world-vs-foot contacts to avoid internal contacts.
                    if (b1 == 0 and b2 == 0) or (b1 != 0 and b2 != 0):
                        continue
                    foot_body_id = b1 if b1 != 0 else b2

                    cforce = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, ic, cforce)
                    f_n = float(cforce[0])  # [N]
                    f_t = float(np.linalg.norm(cforce[1:3]))

                    v6 = np.zeros(6, dtype=np.float64)
                    mujoco.mj_objectVelocity(
                        self.model,
                        self.data,
                        mujoco.mjtObj.mjOBJ_BODY,
                        foot_body_id,
                        v6,
                        0,  # world frame
                    )
                    v_foot_world = v6[3:6]
                    contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                    n_hat = contact_frame[0, :]
                    v_tangent = v_foot_world - np.dot(v_foot_world, n_hat) * n_hat
                    v_t = float(np.linalg.norm(v_tangent))
                    friction_e = f_t * v_t * self.dt_sim  # [J]
                    self._contact_E_friction_step_acc += friction_e
                    # [m/s] signed normal velocity. With MuJoCo's frame[0] normal convention,
                    # use sign-agnostic dissipative power to avoid geom ordering issues.
                    v_n = float(np.dot(v_foot_world, n_hat))  # [m/s]
                    p_normal = -f_n * v_n  # [W]
                    if p_normal > 0.0:
                        self._contact_E_normal_step_acc += p_normal * self.dt_sim  # [J]
                    self._contact_E_step_acc = (
                        self._contact_E_friction_step_acc + self._contact_E_normal_step_acc
                    )
            if hasattr(self, "_ider_on_substep"):
                self._ider_on_substep(joint_targets)
            abs_power_i = self._abs_power_step()
            joint_absP_i = self._joint_abs_power_step()
            joint_posP_i, joint_negP_i = self._joint_power_split_step()
            acc_abs_power += abs_power_i
            acc_joint_absP_8 += joint_absP_i
            acc_joint_posP_8 += joint_posP_i
            acc_joint_negP_8 += joint_negP_i
            # Reward 用本 env step 内 substep 平均：累加 fwd_vel, roll, pitch, base_vx
            fwd_vel_i = -float(self.data.qvel[1])
            acc_fwd_vel += fwd_vel_i
            base_vx_i = float(self.data.qvel[0])
            acc_base_vx += base_vx_i
            base_h_i = float(self.data.qpos[2])
            acc_base_h += base_h_i
            base_vz_i = float(self.data.qvel[2])
            acc_base_vz += base_vz_i
            contacts_i = self._get_contacts_step()
            for leg_i in range(4):
                if float(self._prev_contacts_step[leg_i]) < 0.5 and float(contacts_i[leg_i]) > 0.5:
                    qv_leg = np.array(
                        [float(self.data.qvel[self._joint_dof_adrs[2 * leg_i + k]]) if self._joint_dof_adrs[2 * leg_i + k] >= 0 else 0.0 for k in range(2)],
                        dtype=np.float64,
                    )
                    impact_sum += float(np.sum(qv_leg * qv_leg))
                    impact_count += 1
            self._prev_contacts_step = [float(x) for x in contacts_i]
            qw_i, qx_i, qy_i, qz_i = [float(x) for x in self.data.qpos[3:7]]
            roll_i, pitch_i = self._quat_to_roll_pitch(qw_i, qx_i, qy_i, qz_i)
            acc_roll += roll_i
            acc_pitch += pitch_i
            # CSV: one row per sim step (same granularity as no-substep version)
            if self.enable_csv_log:
                qw, qx, qy, qz = [float(x) for x in self.data.qpos[3:7]]
                roll_i, pitch_i = self._quat_to_roll_pitch(qw, qx, qy, qz)
                joint_Estep_i = joint_absP_i * float(self.dt)
                self._log_step_to_csv(
                    abs_power=float(abs_power_i),
                    ep_index_energy=float(abs_power_i * self.dt),
                    joint_targets=joint_targets,
                    contacts=contacts_i,
                    joint_absP_8=joint_absP_i,
                    joint_Estep_8=joint_Estep_i,
                    roll=float(roll_i),
                    pitch=float(pitch_i),
                    sim_step_override=base_sim + i + 1,
                )
            # Per-sim-step callback (e.g. eval print target vs actual joint angles)
            if self.sim_step_callback is not None:
                actual_qpos_8 = []
                for hip_id, knee_id in self.joint_ids:
                    if hip_id >= 0:
                        actual_qpos_8.append(float(self.data.qpos[int(self.model.jnt_qposadr[hip_id])]))
                    else:
                        actual_qpos_8.append(0.0)
                    if knee_id >= 0:
                        actual_qpos_8.append(float(self.data.qpos[int(self.model.jnt_qposadr[knee_id])]))
                    else:
                        actual_qpos_8.append(0.0)
                self.sim_step_callback(base_sim + i + 1, joint_targets, actual_qpos_8)
            # Per-sim-step 打印各关节电机转矩（可选）
            if self.print_sim_step_torque:
                tau_8 = []
                for hip_act, knee_act in self.actuator_ids:
                    tau_8.append(float(self.data.actuator_force[hip_act]) if hip_act >= 0 else 0.0)
                    tau_8.append(float(self.data.actuator_force[knee_act]) if knee_act >= 0 else 0.0)
                flat_names = []
                for hip_name, knee_name in self.controlled_joint_names:
                    flat_names.append(hip_name)
                    flat_names.append(knee_name)
                sim_step = base_sim + i + 1
                print(f"  [sim_step={sim_step}] 各关节电机转矩 (actuator_force N·m):")
                for idx in range(8):
                    print(f"    index {idx}: {flat_names[idx]} = {tau_8[idx]:+.6f}")
            # Per-sim-step 打印目标关节角度和实际关节角度（可选）
            if self.print_sim_step_joint_angles:
                actual_qpos_8 = []
                for hip_id, knee_id in self.joint_ids:
                    if hip_id >= 0:
                        actual_qpos_8.append(float(self.data.qpos[int(self.model.jnt_qposadr[hip_id])]))
                    else:
                        actual_qpos_8.append(0.0)
                    if knee_id >= 0:
                        actual_qpos_8.append(float(self.data.qpos[int(self.model.jnt_qposadr[knee_id])]))
                    else:
                        actual_qpos_8.append(0.0)
                flat_names = []
                for hip_name, knee_name in self.controlled_joint_names:
                    flat_names.append(hip_name)
                    flat_names.append(knee_name)
                sim_step = base_sim + i + 1
                print(f"  [sim_step={sim_step}] 目标关节角度 (target rad):")
                for leg_i in range(4):
                    q1_des, q2_des = joint_targets[leg_i]
                    print(f"    index {leg_i*2}: {flat_names[leg_i*2]} = {q1_des:+.5f}")
                    print(f"    index {leg_i*2+1}: {flat_names[leg_i*2+1]} = {q2_des:+.5f}")
                print(f"  [sim_step={sim_step}] 实际关节角度 (actual rad):")
                for idx in range(8):
                    print(f"    index {idx}: {flat_names[idx]} = {actual_qpos_8[idx]:+.5f}")
            # 与 train 一致：每个 env step 只 sync 一次（最后一个 substep），避免 eval 比 train 慢 50 倍
            if self._viewer is not None and i == K - 1:
                try:
                    self._viewer.sync()
                except Exception:
                    pass
        # Distance over this macro step
        step_base_y_end = float(self.data.qpos[1])
        step_dy = step_base_y_start - step_base_y_end
        if step_dy > 0.0:
            self._ep_distance += step_dy
        self._last_base_y = step_base_y_end

        self.sim_step_counter += K
        self.control_step_counter += 1

        # Per-substep averages for reward/CSV (energy uses sum over K steps)
        abs_power_step = acc_abs_power / float(K)
        joint_absP_8 = acc_joint_absP_8 / float(K)
        joint_Estep_8 = acc_joint_absP_8 * float(self.dt)  # total work over K steps = sum of (P*dt)
        ep_index_energy = acc_abs_power * float(self.dt)   # total energy this env step
        self._ep_energy += ep_index_energy

        obs = self._get_obs()
        # print(f"Step {self.sim_step_counter}: obs={obs}")   

        # ---------------- Reward：每个 env step 用本步 K 个 substep 的平均状态 ----------------
        mean_fwd_vel = acc_fwd_vel / float(K)
        mean_roll = acc_roll / float(K)
        mean_pitch = acc_pitch / float(K)
        mean_base_vx = acc_base_vx / float(K)
        mean_base_h = acc_base_h / float(K)
        mean_base_vz = acc_base_vz / float(K)

        # r_vel: 与 plot_rewards.py 一致 — 高斯下移再缩放，v=0 时 r_vel=0，v=target 时 r_vel=1，倒退(v<0)保留负值
        vel_err = mean_fwd_vel - self.target_speed
        sigma_vel = float(self.vel_reward_sigma)
        g_v = math.exp(-(vel_err / sigma_vel) ** 2)
        v_ref = max(abs(float(self.target_speed)), self.vel_reward_ref_floor)
        g0 = math.exp(-(v_ref / sigma_vel) ** 2)
        den_vel = max(1.0 - g0, self.vel_reward_den_floor)
        r_vel = (g_v - g0) / den_vel

        pose_err2 = (mean_roll / self.roll_tol) ** 2 + (mean_pitch / self.pitch_tol) ** 2
        r_pose = -(1.0 - math.exp(-min(pose_err2, 10.0)))

        sigma_vx = float(self.xvel_reward_sigma)
        r_xvel = -(1.0 - math.exp(-(mean_base_vx / sigma_vx) ** 2))

        # 供 info / termination 使用：与 reward 一致用平均；termination 仍用最后一帧
        fwd_vel = mean_fwd_vel
        roll = mean_roll
        pitch = mean_pitch
        base_vx = mean_base_vx

        # tau8 (kept)
        tau8 = np.zeros(8, dtype=np.float64)
        idx = 0
        for (hip_act, knee_act) in self.actuator_ids:
            tau8[idx] = float(self.data.actuator_force[int(hip_act)]) if hip_act >= 0 else 0.0
            idx += 1
            tau8[idx] = float(self.data.actuator_force[int(knee_act)]) if knee_act >= 0 else 0.0
            idx += 1

        r_tau = -float(np.mean((tau8 / self.tau_max) ** 2))
        dtau = tau8 - self._last_tau_8
        r_smooth = -float(np.mean(dtau * dtau))  # kept (not in reward by default)
        self._last_tau_8 = tau8
        tau_ratio = np.abs(tau8) / max(float(self.tau_max), 1e-9)
        r_sat = -float(np.mean(np.square(np.maximum(tau_ratio - float(self.sat_threshold), 0.0))))

        joint_posP_step_8 = acc_joint_posP_8 / float(K)
        joint_negP_step_8 = acc_joint_negP_8 / float(K)
        joint_power_pen = float(np.mean(joint_posP_step_8 + self.neg_power_alpha * joint_negP_step_8))
        r_joint_energy = -joint_power_pen
        impact_proxy = float(impact_sum / max(impact_count, 1))
        r_impact = -impact_proxy
        z_ref = float(self.model.qpos0[2])
        r_body = -(
            float((mean_base_h - z_ref) ** 2) * self.body_height_weight
            + float(mean_base_vz * mean_base_vz) * self.body_vz_weight
        )

        # ---------------- Energy bookkeeping (from substep accumulation) ----------------
        # abs_power_step, joint_absP_8, joint_Estep_8, ep_index_energy, _ep_energy, _last_base_y already set above
        dE0 = 0.018
        scale = 0.004
        x = (ep_index_energy - dE0) / scale
        r_energy = -math.tanh(np.log1p(np.exp(x)))

        # termination (kept)
        base_h = float(self.data.qpos[2])
        terminated = bool(base_h < 0.02 or abs(roll) > 0.75 or abs(pitch) > 0.75)
        truncated = bool(self.sim_step_counter >= self.max_episode_steps)
        done = bool(terminated or truncated)

        # stance/swing abs work accumulation per actuator (total over K substeps)
        contacts = self._get_contacts_step()
        abs_work_step_8 = joint_Estep_8

        for a_idx in range(8):
            leg = a_idx // 2
            if leg < 4 and float(contacts[leg]) > 0.5:
                self._ep_stance_abs_work_8[a_idx] += float(abs_work_step_8[a_idx])
            else:
                self._ep_swing_abs_work_8[a_idx] += float(abs_work_step_8[a_idx])

        # CoT (kept)
        cot = 0.0
        if done and (self.cot_weight > 0.0):
            dist = self.model.qpos0[1] - self.data.qpos[1]
            if dist > self.cot_min_dist:
                cot = self._ep_energy / (self.total_mass * self.g * dist)

        ider_reward_delta = 0.0
        if hasattr(self, "_ider_compute_reward_addition"):
            ider_reward_delta = float(self._ider_compute_reward_addition())

        reward = (
            self.vel_weight * float(r_vel)
            + self.pose_weight * float(r_pose)
            + self.x_weight * float(r_xvel)
            + self.sat_weight * float(r_sat)
            + self.joint_energy_weight * float(r_joint_energy)
            + self.impact_weight * float(r_impact)
            + float(self.smooth_weight) * float(r_smooth)
            + float(r_body)
        ) + float(ider_reward_delta)
        if done and self.cot_weight > 0.0 and cot > 0.0:
            reward -= self.cot_weight * min(float(cot), self.cot_clip)

        # episode sums (kept)
        self._ep_total_reward += float(reward)
        self._ep_comp_sums["r_vel"] += float(r_vel)
        self._ep_comp_sums["r_pose"] += float(r_pose)
        self._ep_comp_sums["r_xvel"] += float(r_xvel)
        self._ep_comp_sums["r_sat"] += float(r_sat)
        self._ep_comp_sums["r_joint_energy"] += float(r_joint_energy)
        self._ep_comp_sums["r_impact"] += float(r_impact)
        self._ep_comp_sums["r_body"] += float(r_body)
        self._ep_comp_sums["r_smooth"] += float(r_smooth)

        self._ep_abs_power_sum += float(abs_power_step)
        self._ep_abs_work += float(ep_index_energy)
        self._ep_step_count += 1

        info = {
            "fwd_vel": float(fwd_vel),
            "vel_err": float(vel_err),
            "roll": float(roll),
            "pitch": float(pitch),
            "base_h": float(mean_base_h),
            "base_vx": float(mean_base_vx),
            "base_vy": float(-mean_fwd_vel),
            "base_vz": float(mean_base_vz),
            "base_wx": float(self.data.qvel[3]),
            "base_wy": float(self.data.qvel[4]),
            "base_wz": float(self.data.qvel[5]),
            "contact_E_step": float(self._contact_E_step_acc),
            "contact_E_friction_step": float(self._contact_E_friction_step_acc),
            "contact_E_normal_step": float(self._contact_E_normal_step_acc),
            "abs_power": float(abs_power_step),
            "ep_distance": float(self._ep_distance),
            "cot": float(cot),

            "r_vel": float(r_vel),
            "r_pose": float(r_pose),
            "r_xvel": float(r_xvel),
            "r_tau": float(r_tau),
            "r_energy": float(r_energy),
            "r_sat": float(r_sat),
            "r_joint_energy": float(r_joint_energy),
            "r_impact": float(r_impact),
            "r_body": float(r_body),

            "dy": float(step_dy),
            "contacts": contacts,

            "foot_targets": foot_targets,
            "joint_targets": joint_targets,

            # helpful debug
            "action_f4": self._cur_f4.copy(),
            "action_mu4": self._cur_mu4.copy(),
            # 做法B: 本步实际用于控制的 8 维动作（与 rollout 中应存的动作一致）
            "action_used": (self._last_applied_action.copy() if self._last_applied_action is not None else np.array(action, dtype=np.float64)),
        }
        if hasattr(self, "_ider_extend_step_info"):
            ider_info = self._ider_extend_step_info()
            if isinstance(ider_info, dict):
                info.update(ider_info)
        if self.use_energy_tank and self._energy_tank is not None:
            info["E_tank"] = last_E_tank
            info["alpha_tank"] = last_alpha_tank
            info["P_plus"] = float(self._energy_tank.P_plus_last)
            info["P_minus"] = float(self._energy_tank.P_minus_last)
            info["tau_v_req_8"] = np.asarray(self._energy_tank.tau_v_req_last, dtype=np.float64).copy()
            info["tau_v_eff_8"] = np.asarray(self._energy_tank.tau_v_eff_last, dtype=np.float64).copy()

        if done:
            dist_raw = float(self.model.qpos0[1] - self.data.qpos[1])
            # x 方向偏移（相对初始位置的绝对值，单位 m）
            x_offset_m = float(np.abs(self.data.qpos[0] - self.model.qpos0[0]))
            ep_abs_power_mean = float(self._ep_abs_power_sum / max(self._ep_step_count, 1))

            self._append_episode_summary_row()

            info["episode_components"] = {
                "ep_r_vel": float(self._ep_comp_sums["r_vel"]),
                "ep_r_pose": float(self._ep_comp_sums["r_pose"]),
                "ep_r_xvel": float(self._ep_comp_sums["r_xvel"]),
                "ep_r_sat": float(self._ep_comp_sums["r_sat"]),
                "ep_r_joint_energy": float(self._ep_comp_sums["r_joint_energy"]),
                "ep_r_impact": float(self._ep_comp_sums["r_impact"]),
                "ep_r_body": float(self._ep_comp_sums["r_body"]),
                "ep_r_smooth": float(self._ep_comp_sums["r_smooth"]),
                "ep_abs_work_J": float(self._ep_abs_work),
                "ep_abs_power_mean_W": float(ep_abs_power_mean),
                "ep_energy_J": float(self._ep_energy),
                "ep_distance_pos_dy": float(self._ep_distance),
                "ep_distance_raw": float(dist_raw),
                "x_offset_m": float(x_offset_m),
                "stance_abs_work_total_J": float(np.sum(self._ep_stance_abs_work_8)),
                "swing_abs_work_total_J": float(np.sum(self._ep_swing_abs_work_8)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            if hasattr(self, "_ider_mutate_episode_components"):
                self._ider_mutate_episode_components(info["episode_components"])

            if self.enable_csv_log:
                self._close_csv()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:
        self._close_csv()
        pass
