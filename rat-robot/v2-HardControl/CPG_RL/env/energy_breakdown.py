"""
energy_breakdown.py
===================
Drop-in energy-breakdown analyzer for the rat quadruped CPG sim (ToSim or env).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


class EnergyBreakdownAnalyzer:
    DEFAULT_LEG_NAMES = [
        "thigh_joint_fl", "leg_joint_fl",
        "thigh_joint_fr", "leg_joint_fr",
        "thigh_joint_rl", "leg_joint_rl",
        "thigh_joint_rr", "leg_joint_rr",
    ]
    DEFAULT_TOUCH_NAMES = ["fl_t1", "fr_t1", "rl_t1", "rr_t1"]
    LEG_TAGS = ("fl", "fr", "rl", "rr")

    def __init__(
        self,
        model,
        dt: float,
        tau_max: float = 0.157,
        leg_active_joint_names: Optional[List[str]] = None,
        touch_sensor_names: Optional[List[str]] = None,
        body_y_qpos_index: int = 1,
        forward_sign: float = -1.0,
        record_timeseries: bool = False,
    ):
        if mujoco is None:
            raise RuntimeError("mujoco package is required")

        self.model = model
        self.dt = float(dt)
        self.tau_max = float(tau_max)
        self.body_y_qpos_index = int(body_y_qpos_index)
        self.forward_sign = float(forward_sign)
        self.record_timeseries = bool(record_timeseries)

        leg_names = leg_active_joint_names or list(self.DEFAULT_LEG_NAMES)
        if len(leg_names) != 8:
            raise ValueError("Expected 8 leg joint names (4 legs × 2 joints)")
        self.leg_joint_names = leg_names
        self._setup_active_joint_indices()

        touch_names = touch_sensor_names or list(self.DEFAULT_TOUCH_NAMES)
        self._setup_touch_sensors(touch_names)
        self._leg_all_dofs = self._find_all_leg_dofs()
        self._dof_damping = np.asarray(model.dof_damping, dtype=np.float64).copy()
        self.reset()

    def _setup_active_joint_indices(self):
        info = []
        for jname in self.leg_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in model")
            dof_adr = int(self.model.jnt_dofadr[jid])
            damping = float(self.model.dof_damping[dof_adr])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            info.append({"name": jname, "joint_id": jid, "dof_adr": dof_adr, "actuator_id": int(aid), "damping": damping})
        self.active_joints = info
        self.act_dofs = np.array([j["dof_adr"] for j in info], dtype=int)
        self.act_ids = np.array([j["actuator_id"] for j in info], dtype=int)

    def _setup_touch_sensors(self, touch_names):
        slices = []
        for sname in touch_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid < 0:
                slices.append(None)
                continue
            adr = int(self.model.sensor_adr[sid])
            dim = int(self.model.sensor_dim[sid])
            slices.append((adr, adr + dim))
        self._touch_slices = slices

    def _find_all_leg_dofs(self):
        dofs = []
        for jid in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if any(jname.endswith(f"_{tag}") for tag in self.LEG_TAGS):
                dof_adr = int(self.model.jnt_dofadr[jid])
                dofs.append((jname, dof_adr))
        return dofs

    def reset(self):
        self.n_substeps = 0
        self.W_act_pos = 0.0
        self.W_act_neg = 0.0
        self.W_act_abs_total = 0.0
        self.W_damping_total = 0.0
        self.W_damping_legs = 0.0
        self.W_constraint = 0.0
        self.W_bias_dot_qvel = 0.0
        self.W_saturation_input = 0.0
        self.W_swing_per_leg = np.zeros(4)
        self.W_stance_per_leg = np.zeros(4)
        self.W_swing_late_per_leg = np.zeros(4)
        self.W_stance_late_per_leg = np.zeros(4)
        self.W_swing_in_contact = np.zeros(4)
        self.W_stance_no_contact = np.zeros(4)
        self.sat_count_per_joint = np.zeros(8, dtype=int)
        self._y_start = None
        self._y_end = None
        self._K_start = None
        self._K_last = None

    def update(self, data, leg_phases: Optional[List[float]] = None):
        qvel = np.asarray(data.qvel, dtype=np.float64)
        qfrc_act = np.asarray(data.qfrc_actuator, dtype=np.float64)
        qfrc_con = np.asarray(data.qfrc_constraint, dtype=np.float64)
        qfrc_bias = np.asarray(data.qfrc_bias, dtype=np.float64)
        dt = self.dt
        self.n_substeps += 1

        if self._y_start is None:
            self._y_start = float(data.qpos[self.body_y_qpos_index])
        self._y_end = float(data.qpos[self.body_y_qpos_index])

        K_now = self._compute_kinetic_energy(data)
        if self._K_start is None:
            self._K_start = K_now
        self._K_last = K_now

        P_act_dof = qfrc_act[self.act_dofs] * qvel[self.act_dofs]
        P_act_pos = float(np.maximum(P_act_dof, 0.0).sum())
        P_act_neg = float(np.maximum(-P_act_dof, 0.0).sum())
        P_act_abs = float(np.abs(P_act_dof).sum())
        self.W_act_pos += P_act_pos * dt
        self.W_act_neg += P_act_neg * dt
        self.W_act_abs_total += P_act_abs * dt

        P_damp_all = float((self._dof_damping * qvel * qvel).sum())
        self.W_damping_total += P_damp_all * dt
        P_damp_legs = 0.0
        for _, dof_adr in self._leg_all_dofs:
            d = float(self._dof_damping[dof_adr])
            qd = float(qvel[dof_adr])
            P_damp_legs += d * qd * qd
        self.W_damping_legs += P_damp_legs * dt

        P_con = float((qfrc_con * qvel).sum())
        self.W_constraint += P_con * dt
        P_bias = float((qfrc_bias * qvel).sum())
        self.W_bias_dot_qvel += P_bias * dt

        contacts = self._get_contacts(data)
        for i in range(4):
            P_leg_i = abs(P_act_dof[2 * i] + P_act_dof[2 * i + 1])
            in_contact = bool(contacts[i] > 0.0)
            if leg_phases is not None:
                phase = float(leg_phases[i]) % (2.0 * math.pi)
                is_swing = phase < math.pi
                is_late_swing = (math.pi / 2.0) <= phase < math.pi
                is_late_stance = (1.5 * math.pi) <= phase < (2.0 * math.pi)
            else:
                is_swing = not in_contact
                is_late_swing = False
                is_late_stance = False
            if is_swing:
                self.W_swing_per_leg[i] += P_leg_i * dt
                if is_late_swing:
                    self.W_swing_late_per_leg[i] += P_leg_i * dt
                if in_contact:
                    self.W_swing_in_contact[i] += P_leg_i * dt
            else:
                self.W_stance_per_leg[i] += P_leg_i * dt
                if is_late_stance:
                    self.W_stance_late_per_leg[i] += P_leg_i * dt
                if not in_contact:
                    self.W_stance_no_contact[i] += P_leg_i * dt

        for i in range(8):
            aid = self.act_ids[i]
            if aid < 0:
                continue
            tau = float(data.actuator_force[aid])
            qd = float(qvel[self.act_dofs[i]])
            if abs(tau) >= 0.99 * self.tau_max:
                self.sat_count_per_joint[i] += 1
                self.W_saturation_input += abs(tau * qd) * dt

    def _compute_kinetic_energy(self, data):
        nv = self.model.nv
        Mqv = np.zeros(nv, dtype=np.float64)
        mujoco.mj_mulM(self.model, data, Mqv, np.asarray(data.qvel, dtype=np.float64))
        return 0.5 * float(np.dot(data.qvel, Mqv))

    def _get_contacts(self, data):
        out = np.zeros(4, dtype=float)
        sd = np.asarray(data.sensordata, dtype=np.float64).reshape(-1)
        for i, sl in enumerate(self._touch_slices):
            if sl is None or sl[1] > sd.size:
                out[i] = 0.0
            else:
                a, b = sl
                out[i] = float(np.mean(sd[a:b])) if b > a else 0.0
        return out

    def _design_hints(self):
        W_act = self.W_act_abs_total
        if W_act < 1e-12:
            return {"hints": [{"issue": "no measurable energy yet", "knob": "run longer simulation"}]}
        sw_late = self.W_swing_late_per_leg.sum()
        st_late = self.W_stance_late_per_leg.sum()
        sat_pct = 100.0 * self.W_saturation_input / W_act
        damp_pct = 100.0 * self.W_damping_legs / W_act
        con_pct = 100.0 * (-self.W_constraint) / W_act
        hints = []
        if con_pct > 15.0:
            hints.append({"issue": "Contact dissipation high (>15% W_act).", "knob": "Reduce v_z near touchdown by reshaping swing-late profile."})
        if sw_late > 0.6 * (sw_late + st_late) and sw_late / W_act > 0.20:
            hints.append({"issue": "Late-swing actuator work disproportionately large.", "knob": "Decelerate foot before touchdown."})
        if st_late / W_act > 0.20:
            hints.append({"issue": "Late-stance braking work high.", "knob": "Smooth swing/stance transition."})
        if sat_pct > 5.0:
            hints.append({"issue": f"Actuator saturation energy high ({sat_pct:.1f}%).", "knob": "Reduce a·f² or raise torque margin."})
        if damp_pct > 25.0:
            hints.append({"issue": f"Damping dissipation high ({damp_pct:.1f}%).", "knob": "Reduce qdot peaks via lower f or smoother path."})
        if not hints:
            hints.append({"issue": "No dominant sink", "knob": "Fine-grid search on (a,b,f)."})
        return {"hints": hints}

    def report(self) -> Dict:
        if self.n_substeps == 0:
            return {"error": "no data; call update() first"}
        T = self.n_substeps * self.dt
        delta_y = self._y_end - self._y_start
        distance_forward = self.forward_sign * delta_y
        distance_abs = abs(delta_y)
        delta_K = self._K_last - self._K_start
        delta_PE = -self.W_bias_dot_qvel
        W_damp_diss = self.W_damping_total
        W_con_diss = -self.W_constraint
        W_act_net = self.W_act_pos - self.W_act_neg
        rhs = delta_K + delta_PE + W_damp_diss + W_con_diss
        balance_residual = W_act_net - rhs
        sat_rate = self.sat_count_per_joint / max(self.n_substeps, 1)
        per_leg = {}
        for i, leg_name in enumerate(["FL", "FR", "RL", "RR"]):
            sw = float(self.W_swing_per_leg[i]); st = float(self.W_stance_per_leg[i])
            per_leg[leg_name] = {
                "W_swing_J": sw, "W_stance_J": st,
                "W_swing_late_J": float(self.W_swing_late_per_leg[i]),
                "W_stance_late_J": float(self.W_stance_late_per_leg[i]),
            }
        W_act_abs = self.W_act_abs_total
        return {
            "meta": {"n_substeps": self.n_substeps, "duration_s": T, "dt_s": self.dt, "distance_forward_m": float(distance_forward), "distance_abs_m": float(distance_abs), "forward_sign_convention": self.forward_sign},
            "totals_J": {
                "W_act_abs_input": float(W_act_abs), "W_act_pos": float(self.W_act_pos), "W_act_neg": float(self.W_act_neg), "W_act_net": float(W_act_net),
                "W_damping_dissipated": float(W_damp_diss), "W_constraint_dissipated_est": float(W_con_diss), "W_saturation_segment_input": float(self.W_saturation_input),
                "delta_K": float(delta_K), "delta_PE_via_qfrc_bias": float(delta_PE),
            },
            "ratios_pct_of_W_act_abs": {
                "act_pos": 100.0 * self.W_act_pos / max(W_act_abs, 1e-12),
                "act_neg": 100.0 * self.W_act_neg / max(W_act_abs, 1e-12),
                "damping_legs": 100.0 * self.W_damping_legs / max(W_act_abs, 1e-12),
                "constraint_dissipation": 100.0 * W_con_diss / max(W_act_abs, 1e-12),
                "saturation_segment": 100.0 * self.W_saturation_input / max(W_act_abs, 1e-12),
            },
            "per_leg": per_leg,
            "saturation": {"rate_per_joint": sat_rate.tolist(), "rate_max": float(sat_rate.max()), "rate_mean": float(sat_rate.mean()), "joint_names": list(self.leg_joint_names)},
            "performance": {
                "J_per_m_battery": float(W_act_abs / max(distance_abs, 1e-9)),
                "mean_power_W": float(W_act_abs / max(T, 1e-9)),
                "mean_velocity_mps": float(distance_forward / max(T, 1e-9)),
            },
            "energy_balance_check": {
                "lhs_W_act_net": float(W_act_net), "rhs_dK_plus_dPE_plus_diss": float(rhs), "residual_J": float(balance_residual),
                "residual_pct_of_W_act_net": 100.0 * balance_residual / max(abs(W_act_net), 1e-12),
            },
            "design_hints": self._design_hints(),
        }

    def report_human_readable(self) -> str:
        r = self.report()
        if "error" in r:
            return r["error"]
        lines = ["=" * 72, "ENERGY BREAKDOWN REPORT", "=" * 72]
        m = r["meta"]; p = r["performance"]; t = r["totals_J"]
        lines.append(f"Duration:  {m['duration_s']:.3f} s   ({m['n_substeps']} substeps × {m['dt_s']*1000:.2f} ms)")
        lines.append(f"Distance:  {m['distance_forward_m']*1000:+.2f} mm  (forward, sign={m['forward_sign_convention']:+.0f})")
        lines.append(f"v̄ = {p['mean_velocity_mps']*100:+.2f} cm/s    P̄ = {p['mean_power_W']*1000:.2f} mW    J/m = {p['J_per_m_battery']:.4f} J/m")
        lines.append("Energies (mJ):")
        for k, v in t.items():
            lines.append(f"  {k:<32} {v*1000:+10.4f}")
        return "\n".join(lines)

