# ToSim.py
# -*- coding: utf-8 -*-

import os
import math
import time
import json
import numpy as np
import matplotlib.pyplot as plt

import mujoco as mj
from mujoco import viewer

from energy_utils import EnergyMeter


class SimModel(object):
    """
    悬空测试（方案A） + 能量/数据闭环版本

    本版更新重点（按你的需求）：
    ✅ 不仅记录功率，还记录“每个 dt 的功增量 dW”，默认不做累计（避免叠加做功）
    ✅ 同时保留可选累计（accumulate_energy=True 时启用）
    ✅ 保存：每电机功率/功增量（8维），以及总正功/负功/绝对功的每步增量（标量）
    """

    def __init__(self, modelPath: str):
        super(SimModel, self).__init__()

        self.model = mj.MjModel.from_xml_path(modelPath)
        self.data = mj.MjData(self.model)

        # --------- 解析电机 DOF ---------
        self.motor_joint_names = self._default_motor_joint_names()
        self.motor_dof_idx = self._parse_motor_dof_idx(self.motor_joint_names)

        # 能量统计（与 RL 训练口径一致：使用 motor_dof_idx）
        self.energy_meter = EnergyMeter(
            self.model,
            motor_dof_idx=self.motor_dof_idx,
            ref_site_name="body_ss",
            forward_axis=1,
            use_abs_for_cot=False,
            debug=False,
            debug_every_n_steps=500,
            debug_print_xyz=False,
        )

        self.energy_cum = 0.0
        self.energy_power = []  # 每个 substep 的功率（EnergyMeter 输出）
        self.energy_analyzer = None  # optional EnergyBreakdownAnalyzer

        # actuator -> joint qpos 地址（大步记录误差用）
        self.qerr_log = []
        self.q_cmd_log = []
        self.q_now_log = []
        self._act_qpos_adr = np.zeros(self.model.nu, dtype=int)
        for act_id in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_id][0]
            self._act_qpos_adr[act_id] = self.model.jnt_qposadr[jnt_id]

        # --------- 触觉传感器 slice（不写死 sensordata 下标） ---------
        self.touch_sensor_names = ["fl_t1", "fr_t1", "rl_t1", "rr_t1"]
        self._touch_sensor_slices = self._build_sensor_slices(self.touch_sensor_names)

        # --------- substep 级别日志（关键闭环） ---------
        self.substep_time_log = []          # [T_sub]
        self.actuator_force_log = []        # [T_sub, nu]
        self.motor_qfrc_log = []            # [T_sub, 8]
        self.motor_qvel_log = []            # [T_sub, 8]
        self.ncon_log = []                  # [T_sub]
        self.touch_log = []                 # [T_sub, 4]

        # 解释项（可选但建议）
        self.qfrc_bias_log = []             # [T_sub, nv]
        self.qfrc_passive_log = []          # [T_sub, nv]
        self.qfrc_constraint_log = []       # [T_sub, nv]

        # --------- 功率/功（新增：按 dt 的功增量，不叠加） ---------
        # 每个 substep 的总绝对功率（标量）: Σ|τ*qdot|
        self.motor_power_log = []           # [T_sub]
        self.motor_power_pos_log = []

        # 每个 substep 的每电机有符号功率（8维）: τ*qdot
        self.motor_power_each_log = []      # [T_sub, 8]

        # 每个 substep 的每电机功增量（8维）: (τ*qdot)*dt
        self.motor_work_dt_each_log = []    # [T_sub, 8]

        # 每个 substep 的总功增量（标量），按正/负/abs 分开
        self.motor_work_dt_pos_log = []     # [T_sub]  Σmax(P_i,0)*dt
        self.motor_work_dt_neg_log = []     # [T_sub]  Σmax(-P_i,0)*dt  (吸收功幅值)
        self.motor_work_dt_abs_log = []     # [T_sub]  Σ|P_i|*dt

        # （可选）累计能量：默认不使用
        self.motor_Epos_cum_log = []        # [T_sub]
        self.motor_Eabs_cum_log = []        # [T_sub]
        self._motor_Epos_cum = 0.0
        self._motor_Eabs_cum = 0.0

        # --------- 每步打印：关节误差 + 关节力矩（测试用） ---------
        self.print_step_joint_error_torque = False

        # --------- 渲染 ---------
        self.render_enabled = False
        self._viewer_ctx = None
        self._viewer_cam = (0.6, -15.0, 90.0)  # (distance_factor, elevation, azimuth)
        self._last_render_sync = time.time()

        # --------- 轨迹记录（保持你原来的接口） ---------
        self.legPosName = [
            ["thigh_link_fl", "ankle_fl"],
            ["thigh_link_fr", "ankle_fr"],
            ["thigh_link_rl", "ankle_rl"],
            ["thigh_link_rr", "ankle_rr"]
        ]
        self.fixPoint = "body_ss"

        self._site_fix_id = self._name2id(mj.mjtObj.mjOBJ_SITE, self.fixPoint)
        self._site_leg_ids = []
        for pair in self.legPosName:
            sid0 = self._name2id(mj.mjtObj.mjOBJ_SITE, pair[0])
            sid1 = self._name2id(mj.mjtObj.mjOBJ_SITE, pair[1])
            self._site_leg_ids.append((sid0, sid1))

        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]
        self.movePath = [[], [], []]
        self.body_velocity = []
        self.feet_world_z = [[], [], [], []]
        self._prev_fix_pos = None

        mj.mj_forward(self.model, self.data)

        # 运行时打印执行器参数（kp/kv、forcerange），便于核对为何误差>0.00157 时力矩未打满
        self._print_actuator_params()

    # -------------------- motor joint naming --------------------
    @staticmethod
    def _default_motor_joint_names():
        suffix = ["fl", "fr", "rl", "rr"]
        names = []
        for s in suffix:
            names.append(f"leg_joint_{s}")
            names.append(f"thigh_joint_{s}")
        return names

    def _parse_motor_dof_idx(self, joint_names):
        dof_idx = []
        missing = []
        for jname in joint_names:
            j_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
            if j_id < 0:
                missing.append(jname)
                continue
            adr = int(self.model.jnt_dofadr[j_id])
            dof_idx.append(adr)
        if len(missing) > 0:
            print("[WARN] Missing motor joints in model:", missing)
        if len(dof_idx) == 0:
            raise RuntimeError("Failed to parse motor dof indices: no joints found.")
        return dof_idx

    # -------------------- sensor slices --------------------
    def _build_sensor_slices(self, sensor_names):
        slices = {}
        for sname in sensor_names:
            sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, sname)
            if sid < 0:
                slices[sname] = None
                print(f"[WARN] sensor '{sname}' not found in model.")
                continue
            adr = int(self.model.sensor_adr[sid])
            dim = int(self.model.sensor_dim[sid])
            slices[sname] = (adr, adr + dim)
        return slices

    # -------------------- utils --------------------
    def _name2id(self, obj_type, name):
        try:
            return mj.mj_name2id(self.model, obj_type, name)
        except Exception:
            return -1

    def _print_actuator_params(self):
        """运行时打印执行器 kp/kv 与 forcerange（position 执行器：gainprm[0]=kp, biasprm[1]=-kv）。"""
        nu = self.model.nu
        if nu == 0:
            return
        gain = getattr(self.model, "actuator_gainprm", None)
        bias = getattr(self.model, "actuator_biasprm", None)
        fr = getattr(self.model, "actuator_forcerange", None)
        print("[ToSim] 执行器参数 (运行时 model.actuator_*):")
        for i in range(nu):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            kp = float(gain[i, 0]) if gain is not None and gain.ndim >= 2 else (float(gain[i]) if gain is not None else float("nan"))
            kv = -float(bias[i, 1]) if (bias is not None and bias.ndim >= 2 and bias.shape[1] > 1) else float("nan")
            if fr is not None and fr.ndim >= 2:
                fmin, fmax = float(fr[i, 0]), float(fr[i, 1])
            else:
                fmin = fmax = float("nan")
            print(f"  [{i}] {name}: kp={kp}, kv={kv}, forcerange=[{fmin}, {fmax}]")
        print("  (position 执行器: tau ≈ kp*(q_cmd-q) - kv*qvel, 故 kv>0 时误差大但力矩可未饱和)")

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def enable_render(self, flag=True, camera=(0.6, -15.0, 90.0)):
        self.render_enabled = bool(flag)
        self._viewer_cam = camera
        if (not self.render_enabled) and (self._viewer_ctx is not None):
            try:
                self._viewer_ctx.close()
            except Exception:
                pass
            self._viewer_ctx = None

    def _ensure_viewer(self):
        if not self.render_enabled:
            return
        if self._viewer_ctx is None:
            self._viewer_ctx = viewer.launch_passive(self.model, self.data)
            dist, elev, azim = self._viewer_cam
            self._viewer_ctx.cam.distance = self.model.stat.extent * dist
            self._viewer_ctx.cam.elevation = elev
            self._viewer_ctx.cam.azimuth = azim
            self._last_render_sync = time.time()

    # -------------------- reset logs --------------------
    def initializing(self):
        self.movePath = [[], [], []]
        self.legRealPoint_x = [[], [], [], []]
        self.legRealPoint_y = [[], [], [], []]
        self.body_velocity = []
        self.feet_world_z = [[], [], [], []]

        self.energy_cum = 0.0
        self.energy_power = []
        self.energy_meter.reset()

        self.qerr_log = []
        self.q_cmd_log = []
        self.q_now_log = []

        self.substep_time_log = []
        self.actuator_force_log = []
        self.motor_qfrc_log = []
        self.motor_qvel_log = []
        self.ncon_log = []
        self.touch_log = []

        self.qfrc_bias_log = []
        self.qfrc_passive_log = []
        self.qfrc_constraint_log = []

        # power/work logs
        self.motor_power_log = []
        self.motor_power_pos_log = []
        self.motor_power_each_log = []
        self.motor_work_dt_each_log = []
        self.motor_work_dt_pos_log = []
        self.motor_work_dt_neg_log = []
        self.motor_work_dt_abs_log = []

        # optional cumulative
        self.motor_Epos_cum_log = []
        self.motor_Eabs_cum_log = []
        self._motor_Epos_cum = 0.0
        self._motor_Eabs_cum = 0.0

        if self._site_fix_id >= 0:
            self._prev_fix_pos = self.data.site_xpos[self._site_fix_id].copy()
        else:
            self._prev_fix_pos = None

    def print_action(self):
        print(f"time={self.data.time:.4f}, nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        n_show = min(12, self.model.nq)
        print("qpos[:{}] = {}".format(n_show, np.array(self.data.qpos[:n_show])))

    def getTime(self):
        return float(self.data.time)

    # -------------------- simulation step --------------------
    def runStep(self, ctrlData, cur_time_step, realtime=False, accumulate_energy=False):
        """
        accumulate_energy:
          False(默认): 不累计功，只记录每个 dt 的 dW（你要的）
          True: 同时更新 motor_Epos_cum / motor_Eabs_cum（可选）
        """
        self._ensure_viewer()

        ctrlData = np.asarray(ctrlData, dtype=float)
        assert ctrlData.shape[0] == self.model.nu, f"ctrlData length {ctrlData.shape[0]} != nu {self.model.nu}"
        self.data.ctrl[:] = ctrlData

        sim_dt = float(self.model.opt.timestep)
        step_num = max(1, int(round(cur_time_step / sim_dt)))

        for _ in range(step_num):
            mj.mj_step(self.model, self.data)
            if self.energy_analyzer is not None:
                # Optional plug-in point for per-substep energy breakdown.
                self.energy_analyzer.update(self.data, leg_phases=None)
            # print(self.data.qvel)
            # --- EnergyMeter (RL-aligned) ---
            P_em, E_inc = self.energy_meter.update(self.data, sim_dt)
            self.energy_cum += E_inc
            self.energy_power.append(float(P_em))

            # --- substep time ---
            self.substep_time_log.append(float(self.data.time))

            # --- actuator output force (nu) ---
            if self.data.actuator_force is not None and self.data.actuator_force.size == self.model.nu:
                self.actuator_force_log.append(self.data.actuator_force.copy())
            else:
                self.actuator_force_log.append(np.full((self.model.nu,), np.nan, dtype=float))

            # --- motor qfrc / qvel (8) ---
            if self.data.qfrc_actuator is not None and self.data.qfrc_actuator.size == self.model.nv:
                mot_tau = self.data.qfrc_actuator[self.motor_dof_idx].copy()
            else:
                mot_tau = np.full((len(self.motor_dof_idx),), np.nan, dtype=float)
            self.motor_qfrc_log.append(mot_tau)

            if self.data.qvel is not None and self.data.qvel.size == self.model.nv:
                mot_qv = self.data.qvel[self.motor_dof_idx].copy()
            else:
                mot_qv = np.full((len(self.motor_dof_idx),), np.nan, dtype=float)
            self.motor_qvel_log.append(mot_qv)

            # ---------------------------------------------------------
            # 关键：每 dt 的功率/功增量（不做累计，直接存 dW）
            # ---------------------------------------------------------
            P_each = mot_tau * mot_qv  # [8] signed mechanical power per motor
            # print(P_each)
            P_pos = float(np.nansum(np.maximum(P_each, 0.0)))      # Σ pos power
            # print(P_pos)
            P_neg = float(np.nansum(np.maximum(-P_each, 0.0)))     # Σ neg power magnitude
            P_abs = float(np.nansum(np.abs(P_each)))               # Σ abs power

            # 每 dt 的功增量（J）
            dW_each = P_each * sim_dt                               # [8] signed
            dW_pos = P_pos * sim_dt                                 # scalar
            dW_neg = P_neg * sim_dt                                 # scalar (absorption)
            dW_abs = P_abs * sim_dt                                 # scalar

            # 日志保存
            self.motor_power_each_log.append(P_each.copy())
            self.motor_power_pos_log.append(P_pos)
            self.motor_power_log.append(P_abs)                      # 保持你原来的“总功率”曲线接口：用 abs 总和
            self.motor_work_dt_each_log.append(dW_each.copy())
            self.motor_work_dt_pos_log.append(dW_pos)
            self.motor_work_dt_neg_log.append(dW_neg)
            self.motor_work_dt_abs_log.append(dW_abs)

            # 可选累计（如果你还想要累计曲线）
            if accumulate_energy:
                self._motor_Epos_cum += dW_pos
                self._motor_Eabs_cum += dW_abs
                self.motor_Epos_cum_log.append(self._motor_Epos_cum)
                self.motor_Eabs_cum_log.append(self._motor_Eabs_cum)

            # --- contact count ---
            self.ncon_log.append(int(self.data.ncon))

            # --- touch sensors (4) ---
            sens = np.asarray(self.data.sensordata)
            touch_vec = []
            for sname in self.touch_sensor_names:
                sl = self._touch_sensor_slices.get(sname, None)
                if sl is None:
                    touch_vec.append(np.nan)
                else:
                    a, b = sl
                    touch_vec.append(float(np.mean(sens[a:b])))
            self.touch_log.append(np.array(touch_vec, dtype=float))

            # --- dynamics breakdown (nv) ---
            self.qfrc_bias_log.append(self.data.qfrc_bias.copy())
            self.qfrc_passive_log.append(self.data.qfrc_passive.copy())
            self.qfrc_constraint_log.append(self.data.qfrc_constraint.copy())

        # --- big-step path/velocity ---
        if self._site_fix_id >= 0:
            fix_pos = self.data.site_xpos[self._site_fix_id].copy()
            for i in range(3):
                self.movePath[i].append(fix_pos[i])

            if self._prev_fix_pos is not None:
                dy = fix_pos[1] - self._prev_fix_pos[1]
                vy = dy / (step_num * sim_dt)
                self.body_velocity.append(float(vy))
            else:
                self.body_velocity.append(0.0)
            self._prev_fix_pos = fix_pos.copy()
        else:
            self.body_velocity.append(0.0)

        # --- legs y/z + feet zabs (big-step) ---
        for i, (sid_origin, sid_ankle) in enumerate(self._site_leg_ids):
            if sid_origin >= 0 and sid_ankle >= 0:
                originPoint = self.data.site_xpos[sid_origin]
                currentPoint = self.data.site_xpos[sid_ankle]
                tYrel = currentPoint[1] - originPoint[1]
                tZrel = currentPoint[2] - originPoint[2]
                zAbs = currentPoint[2]
            else:
                tYrel, tZrel, zAbs = np.nan, np.nan, np.nan
            self.legRealPoint_x[i].append(float(tYrel))
            self.legRealPoint_y[i].append(float(tZrel))
            self.feet_world_z[i].append(float(zAbs))

        # --- big-step actuator->joint error ---
        q_cmd = ctrlData
        q_now = np.zeros_like(q_cmd)
        for act_id in range(self.model.nu):
            qpos_idx = self._act_qpos_adr[act_id]
            q_now[act_id] = self.data.qpos[qpos_idx]
        q_err = np.abs(q_cmd - q_now)
        self.qerr_log.append(q_err)
        self.q_cmd_log.append(q_cmd.copy())
        self.q_now_log.append(q_now)

        # --- 每步打印关节误差和关节力矩（测试用） ---
        if self.print_step_joint_error_torque and len(self.actuator_force_log) > 0:
            step_idx = len(self.qerr_log)
            tau = self.actuator_force_log[-1]
            print(f"[step {step_idx}] 关节误差 (rad) | 关节力矩 (N·m)")
            for j, name in enumerate(self.motor_joint_names):
                print(f"  {name}: err={q_err[j]:.6f}  tau={tau[j]:+.6f}")
            print(f"  (q_cmd={np.array2string(q_cmd, precision=4)}, q_now={np.array2string(q_now, precision=4)})")

        # --- render sync ---
        if self.render_enabled and self._viewer_ctx is not None:
            self._viewer_ctx.sync()
            if realtime:
                now = time.time()
                remain = sim_dt - (now - self._last_render_sync)
                if remain > 0:
                    time.sleep(remain)
                self._last_render_sync = now

    # -------------------- path helpers --------------------
    @staticmethod
    def point_distance_line(point, line_point1, line_point2):
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        return np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)

    def drawPath(self):
        path_X = self.movePath[0]
        path_Y = self.movePath[1]
        tL = len(path_X)
        if tL < 2:
            print("Path too short.")
            return 0.0

        dX = path_X[0] - path_X[-1]
        dY = path_Y[0] - path_Y[-1]
        dis = math.sqrt(dX * dX + dY * dY)
        print("Dis --> ", dis)

        start_p = np.array([path_X[0], path_Y[0]])
        end_p = np.array([path_X[-1], path_Y[-1]])
        maxDis = 0.0
        for i in range(tL):
            cur_p = np.array([path_X[i], path_Y[i]])
            tDis = self.point_distance_line(cur_p, start_p, end_p)
            if tDis > maxDis:
                maxDis = tDis
        print("MaxDiff --> ", maxDis)

        plt.figure()
        plt.plot(path_X, path_Y)
        plt.axis('equal')
        plt.grid()
        plt.show()

        new_len = 2000
        orig_len = len(path_X)
        new_idx = np.linspace(0, orig_len - 1, new_len)
        downsampled_X = np.interp(new_idx, np.arange(orig_len), path_X)
        downsampled_Y = np.interp(new_idx, np.arange(orig_len), path_Y)
        path = np.stack((downsampled_X, downsampled_Y), axis=1)
        np.save("path_xy.npy", path)

        return dis

    # -------------------- export --------------------
    def save_numpy(self, out_dir="sim_logs"):
        os.makedirs(out_dir, exist_ok=True)

        # ---- big-step ----
        np.save(os.path.join(out_dir, "path_time.npy"), np.arange(len(self.movePath[0])) * self.dt)
        np.save(os.path.join(out_dir, "path_xyz.npy"), np.stack(self.movePath, axis=1))  # [T_big, 3]
        np.save(os.path.join(out_dir, "body_vy.npy"), np.asarray(self.body_velocity, dtype=float))

        legs_yrel = [np.asarray(v, dtype=float) for v in self.legRealPoint_x]
        legs_zrel = [np.asarray(v, dtype=float) for v in self.legRealPoint_y]
        legs_zabs = [np.asarray(v, dtype=float) for v in self.feet_world_z]
        np.save(os.path.join(out_dir, "feet_yrel.npy"), np.stack(legs_yrel, axis=0))  # [4, T_big]
        np.save(os.path.join(out_dir, "feet_zrel.npy"), np.stack(legs_zrel, axis=0))  # [4, T_big]
        np.save(os.path.join(out_dir, "feet_zabs.npy"), np.stack(legs_zabs, axis=0))  # [4, T_big]

        # EnergyMeter
        np.save(os.path.join(out_dir, "energy_power.npy"), np.asarray(self.energy_power, dtype=float))
        np.save(os.path.join(out_dir, "energy_cum.npy"), np.asarray(self.energy_cum, dtype=np.float64))

        # q cmd/now/err (big-step)
        if len(self.qerr_log) > 0:
            q_err_arr = np.stack(self.qerr_log, axis=0)
            np.save(os.path.join(out_dir, "q_err.npy"), q_err_arr)
            np.savetxt(os.path.join(out_dir, "q_err.csv"), q_err_arr, delimiter=",")
        if len(self.q_cmd_log) > 0:
            q_cmd_arr = np.stack(self.q_cmd_log, axis=0)
            np.save(os.path.join(out_dir, "q_cmd.npy"), q_cmd_arr)
            np.savetxt(os.path.join(out_dir, "q_cmd.csv"), q_cmd_arr, delimiter=",")
        if len(self.q_now_log) > 0:
            q_now_arr = np.stack(self.q_now_log, axis=0)
            np.save(os.path.join(out_dir, "q_now.npy"), q_now_arr)
            np.savetxt(os.path.join(out_dir, "q_now.csv"), q_now_arr, delimiter=",")

        # ---- substep ----
        if len(self.substep_time_log) > 0:
            t_sub = np.asarray(self.substep_time_log, dtype=np.float64)
            np.save(os.path.join(out_dir, "substep_time.npy"), t_sub)
            np.savetxt(os.path.join(out_dir, "substep_time.csv"), t_sub.reshape(-1, 1), delimiter=",")

        if len(self.actuator_force_log) > 0:
            act_f = np.stack(self.actuator_force_log, axis=0)
            np.save(os.path.join(out_dir, "actuator_force_substep.npy"), act_f)
            np.savetxt(os.path.join(out_dir, "actuator_force_substep.csv"), act_f, delimiter=",")

        if len(self.motor_qfrc_log) > 0:
            mot_tau = np.stack(self.motor_qfrc_log, axis=0)
            np.save(os.path.join(out_dir, "motor_qfrc_substep.npy"), mot_tau)
            np.savetxt(os.path.join(out_dir, "motor_qfrc_substep.csv"), mot_tau, delimiter=",")

        if len(self.motor_qvel_log) > 0:
            mot_qv = np.stack(self.motor_qvel_log, axis=0)
            np.save(os.path.join(out_dir, "motor_qvel_substep.npy"), mot_qv)
            np.savetxt(os.path.join(out_dir, "motor_qvel_substep.csv"), mot_qv, delimiter=",")

        if len(self.ncon_log) > 0:
            ncon = np.asarray(self.ncon_log, dtype=int)
            np.save(os.path.join(out_dir, "ncon_substep.npy"), ncon)
            np.savetxt(os.path.join(out_dir, "ncon_substep.csv"), ncon.reshape(-1, 1), delimiter=",")

        if len(self.touch_log) > 0:
            touch = np.stack(self.touch_log, axis=0)
            np.save(os.path.join(out_dir, "touch_substep.npy"), touch)
            np.savetxt(os.path.join(out_dir, "touch_substep.csv"), touch, delimiter=",")

        if len(self.qfrc_bias_log) > 0:
            bias = np.stack(self.qfrc_bias_log, axis=0)
            np.save(os.path.join(out_dir, "qfrc_bias_substep.npy"), bias)
            # 不建议存 csv（太大）；如确实要可开
            np.savetxt(os.path.join(out_dir, "qfrc_bias_substep.csv"), bias, delimiter=",")

        if len(self.qfrc_passive_log) > 0:
            pas = np.stack(self.qfrc_passive_log, axis=0)
            np.save(os.path.join(out_dir, "qfrc_passive_substep.npy"), pas)
            np.savetxt(os.path.join(out_dir, "qfrc_passive_substep.csv"), pas, delimiter=",")

        if len(self.qfrc_constraint_log) > 0:
            con = np.stack(self.qfrc_constraint_log, axis=0)
            np.save(os.path.join(out_dir, "qfrc_constraint_substep.npy"), con)
            np.savetxt(os.path.join(out_dir, "qfrc_constraint_substep.csv"), con, delimiter=",")

        # --- power/workdt outputs ---
        if len(self.motor_power_log) > 0:
            p = np.asarray(self.motor_power_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_power_abs_sum_substep.npy"), p)
            np.savetxt(os.path.join(out_dir, "motor_power_abs_sum_substep.csv"), p.reshape(-1, 1), delimiter=",")
            
                # --- power/workdt outputs ---
        if len(self.motor_power_pos_log) > 0:
            posp = np.asarray(self.motor_power_pos_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_power_pos_sum_substep.npy"), posp)
            np.savetxt(os.path.join(out_dir, "motor_power_pos_sum_substep.csv"), posp.reshape(-1, 1), delimiter=",")

        if len(self.motor_power_each_log) > 0:
            pe = np.stack(self.motor_power_each_log, axis=0)  # [T,8]
            np.save(os.path.join(out_dir, "motor_power_each_substep.npy"), pe)
            np.savetxt(os.path.join(out_dir, "motor_power_each_substep.csv"), pe, delimiter=",")

        if len(self.motor_work_dt_each_log) > 0:
            dwe = np.stack(self.motor_work_dt_each_log, axis=0)  # [T,8]
            np.save(os.path.join(out_dir, "motor_work_dt_each_substep.npy"), dwe)
            np.savetxt(os.path.join(out_dir, "motor_work_dt_each_substep.csv"), dwe, delimiter=",")

        if len(self.motor_work_dt_pos_log) > 0:
            dwp = np.asarray(self.motor_work_dt_pos_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_work_dt_pos_substep.npy"), dwp)
            np.savetxt(os.path.join(out_dir, "motor_work_dt_pos_substep.csv"), dwp.reshape(-1, 1), delimiter=",")

        if len(self.motor_work_dt_neg_log) > 0:
            dwn = np.asarray(self.motor_work_dt_neg_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_work_dt_neg_substep.npy"), dwn)
            np.savetxt(os.path.join(out_dir, "motor_work_dt_neg_substep.csv"), dwn.reshape(-1, 1), delimiter=",")

        if len(self.motor_work_dt_abs_log) > 0:
            dwa = np.asarray(self.motor_work_dt_abs_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_work_dt_abs_substep.npy"), dwa)
            np.savetxt(os.path.join(out_dir, "motor_work_dt_abs_substep.csv"), dwa.reshape(-1, 1), delimiter=",")

        # optional cumulative (only if accumulate_energy=True when running)
        if len(self.motor_Epos_cum_log) > 0:
            epos = np.asarray(self.motor_Epos_cum_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_Epos_cum_substep.npy"), epos)
            np.savetxt(os.path.join(out_dir, "motor_Epos_cum_substep.csv"), epos.reshape(-1, 1), delimiter=",")

        if len(self.motor_Eabs_cum_log) > 0:
            eabs = np.asarray(self.motor_Eabs_cum_log, dtype=float)
            np.save(os.path.join(out_dir, "motor_Eabs_cum_substep.npy"), eabs)
            np.savetxt(os.path.join(out_dir, "motor_Eabs_cum_substep.csv"), eabs.reshape(-1, 1), delimiter=",")

        # meta
        meta = {
            "motor_joint_names": self.motor_joint_names,
            "motor_dof_idx": self.motor_dof_idx,
            "touch_sensor_names": self.touch_sensor_names,
            "dt": self.dt,
            "nu": int(self.model.nu),
            "nv": int(self.model.nv),
            "nq": int(self.model.nq),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[SimModel] Saved logs → {out_dir}")

    def savePath(self, flag):
        file_dir = "Data"
        os.makedirs(file_dir, exist_ok=True)
        filePath = os.path.join(file_dir, "path_" + flag + ".txt")
        with open(filePath, 'w') as f:
            dL = len(self.movePath[0])
            for i in range(dL):
                for j in range(3):
                    f.write(str(self.movePath[j][i]) + ' ')
                f.write('\n')

    def get_feet_condition(self):
        sens = np.array(self.data.sensordata)
        return sens.copy()

    def cot(self) -> float:
        return float(self.energy_meter.cot())
