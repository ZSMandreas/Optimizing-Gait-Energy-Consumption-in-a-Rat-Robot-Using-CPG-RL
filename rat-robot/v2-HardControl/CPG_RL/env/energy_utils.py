# energy_utils.py
# -*- coding: utf-8 -*-
import numpy as np

G = 9.81


class EnergyMeter:
    def __init__(
        self,
        model,
        mass=None,
        motor_dof_idx=None,
        ref_site_name="body_ss",
        forward_axis=1,             # 0:x 1:y 2:z
        use_abs_for_cot=False,      # False: use E_pos for CoT; True: use E_abs for CoT
        store_per_dof_power=False,

        # === debug options ===
        debug=False,
        debug_every_n_steps=500,    # print once every N update() calls
        debug_print_xyz=True,       # print xyz progress when debug=True
    ):
        self.mass = float(np.sum(model.body_mass[1:])) if mass is None else float(mass)
        self.idx = np.array(motor_dof_idx, dtype=int) if motor_dof_idx is not None else None

        self.ref_site_name = ref_site_name
        self.forward_axis = int(forward_axis)
        self.use_abs_for_cot = bool(use_abs_for_cot)
        self.store_per_dof_power = bool(store_per_dof_power)

        # debug
        self.debug = bool(debug)
        self.debug_every_n_steps = int(debug_every_n_steps)
        self.debug_print_xyz = bool(debug_print_xyz)
        self._step_counter = 0

        # resolve site id once
        try:
            import mujoco as mj
            self._mj = mj
            self.ref_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ref_site_name)
        except Exception:
            self._mj = None
            self.ref_site_id = -1

        self.reset()

    def reset(self):
        # energy accumulators
        self.E_pos = 0.0
        self.E_abs = 0.0

        # 1D progress (forward axis)
        self.s0 = None
        self.last_s = None

        # 3D progress (xyz)
        self.p0 = None          # np.array([x0,y0,z0])
        self.last_p = None      # np.array([x,y,z])

        # last-step diagnostics
        self.last_power_pos = 0.0
        self.last_power_abs = 0.0
        self.last_e_inc_pos = 0.0
        self.last_e_inc_abs = 0.0

        self.last_p_each = None
        self._step_counter = 0

    def _get_progress_1d(self, data):
        """Progress along forward_axis using ref site."""
        if self.ref_site_id < 0:
            return 0.0
        return float(data.site_xpos[self.ref_site_id][self.forward_axis])

    def _get_progress_xyz(self, data):
        """World xyz position of ref site."""
        if self.ref_site_id < 0:
            return None
        p = np.array(data.site_xpos[self.ref_site_id], dtype=float)  # (3,)
        return p

    def displacement(self):
        """Absolute displacement along forward axis."""
        if self.s0 is None or self.last_s is None:
            return 0.0
        return float(abs(self.last_s - self.s0))

    def displacement_xyz(self):
        """Returns (dx,dy,dz, d_norm)."""
        if self.p0 is None or self.last_p is None:
            return 0.0, 0.0, 0.0, 0.0
        dp = self.last_p - self.p0
        dnorm = float(np.linalg.norm(dp))
        return float(dp[0]), float(dp[1]), float(dp[2]), dnorm

    def update(self, data, dt):
        """
        Call after each mj_step.
        Computes both:
          P_pos = sum(max(tau*qd, 0))
          P_abs = sum(abs(tau*qd))
        and integrates them.
        """
        self._step_counter += 1

        tau_dof = data.qfrc_actuator
        qd = data.qvel

        if self.idx is None:
            sel_tau = tau_dof
            sel_qd = qd
        else:
            sel_tau = tau_dof[self.idx]
            sel_qd = qd[self.idx]

        p_each = sel_tau * sel_qd

        P_pos = float(np.sum(np.maximum(p_each, 0.0)))
        P_abs = float(np.sum(np.abs(p_each)))

        E_inc_pos = P_pos * float(dt)
        E_inc_abs = P_abs * float(dt)

        self.E_pos += E_inc_pos
        self.E_abs += E_inc_abs

        self.last_power_pos = P_pos
        self.last_power_abs = P_abs
        self.last_e_inc_pos = E_inc_pos
        self.last_e_inc_abs = E_inc_abs

        if self.store_per_dof_power:
            self.last_p_each = p_each.copy()
        else:
            self.last_p_each = None

        # progress update (1D)
        s = self._get_progress_1d(data)
        if self.s0 is None:
            self.s0 = s
        self.last_s = s

        # progress update (xyz)
        p = self._get_progress_xyz(data)
        if p is not None:
            if self.p0 is None:
                self.p0 = p.copy()
            self.last_p = p.copy()

        # === debug print (throttled) ===
        if self.debug and (self._step_counter % self.debug_every_n_steps == 0):
            d = self.displacement()
            dx, dy, dz, dxyz = self.displacement_xyz()

            E_used = self.E_abs if self.use_abs_for_cot else self.E_pos
            cot = E_used / (self.mass * G * max(d, 1e-6))

            if self.debug_print_xyz and (self.last_p is not None):
                x, y, z = self.last_p
                x0, y0, z0 = self.p0 if self.p0 is not None else (np.nan, np.nan, np.nan)
                print(
                    f"[EnergyMeter] step={self._step_counter} t={float(data.time):.3f} "
                    f"pos=({x:.6f},{y:.6f},{z:.6f}) "
                    f"pos0=({x0:.6f},{y0:.6f},{z0:.6f}) "
                    f"dp=({dx:.9f},{dy:.9f},{dz:.9f}) |dp|={dxyz:.9f} "
                    f"s(axis={self.forward_axis})={s:.6f} d_axis={d:.12f} "
                    f"P_pos={P_pos:.6f} P_abs={P_abs:.6f} "
                    f"E_pos={self.E_pos:.6f} E_abs={self.E_abs:.6f} "
                    f"CoT({'abs' if self.use_abs_for_cot else 'pos'})={cot:.6f}"
                )
            else:
                print(
                    f"[EnergyMeter] step={self._step_counter} t={float(data.time):.3f} "
                    f"s={s:.6f} d={d:.12f} "
                    f"P_pos={P_pos:.6f} P_abs={P_abs:.6f} "
                    f"E_pos={self.E_pos:.6f} E_abs={self.E_abs:.6f} "
                    f"CoT({'abs' if self.use_abs_for_cot else 'pos'})={cot:.6f}"
                )

        # return power definition used externally
        if self.use_abs_for_cot:
            return P_abs, E_inc_abs
        else:
            return P_pos, E_inc_pos

    def cot(self):
        d = max(self.displacement(), 1e-6)
        E = self.E_abs if self.use_abs_for_cot else self.E_pos
        cot_val = float(E / (self.mass * G * d))

        if self.debug:
            dx, dy, dz, dxyz = self.displacement_xyz()
            print(
                f"[EnergyMeter:FINAL] s0={self.s0:.6f} s_last={self.last_s:.6f} "
                f"d_axis={self.displacement():.12f} "
                f"dp=({dx:.9f},{dy:.9f},{dz:.9f}) |dp|={dxyz:.9f} "
                f"E_pos={self.E_pos:.6f} E_abs={self.E_abs:.6f} "
                f"CoT({'abs' if self.use_abs_for_cot else 'pos'})={cot_val:.6f}"
            )

        return cot_val
