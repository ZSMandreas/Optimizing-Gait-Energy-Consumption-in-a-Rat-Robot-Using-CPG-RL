# LegModel/forPath.py
# -*- coding: utf-8 -*-

import math
import numpy as np


class CPGNetwork4:
    """
    4-leg coupled oscillator network (polar form):
      r_dot     = alpha*(mu - r^2)*r
      theta_dot = omega + sum_j K_ij * sin(theta_j - theta_i - phi_ij)

    leg order: 0 FL, 1 FR, 2 HL, 3 HR
    desired_offsets: desired phase for each leg relative to an arbitrary reference.
      For trot: [0, pi, pi, 0]
    """

    def __init__(
        self,
        fre_cyc: float,
        dt: float,
        desired_offsets,
        alpha: float = 30.0,
        mu: float = 1.0,
        k: float = 8.0,
        init_thetas=None,
        init_rs=None,
    ):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.k = float(k)

        self.fre_cyc = float(fre_cyc)
        self.omega = 2.0 * math.pi * self.fre_cyc

        self.N = 4

        desired_offsets = np.array(desired_offsets, dtype=float).reshape(-1)
        assert desired_offsets.shape[0] == 4, "desired_offsets must have length 4"
        self.desired = desired_offsets % (2.0 * math.pi)

        # phi_ij: desired (theta_j - theta_i)
        # If we want (theta_j - theta_i) -> (desired_j - desired_i)
        self.phi = (self.desired.reshape(1, -1) - self.desired.reshape(-1, 1))  # [i,j]

        # states
        if init_thetas is None:
            # start near desired offsets
            init_thetas = self.desired.copy()
        if init_rs is None:
            init_rs = np.ones(self.N, dtype=float) * math.sqrt(self.mu)

        self.theta = np.array(init_thetas, dtype=float) % (2.0 * math.pi)
        self.r = np.array(init_rs, dtype=float)

        # coupling matrix (all-to-all, no self coupling)
        self.K = np.ones((self.N, self.N), dtype=float) * self.k
        np.fill_diagonal(self.K, 0.0)

    def set_freq(self, fre_cyc: float):
        self.fre_cyc = float(fre_cyc)
        self.omega = 2.0 * math.pi * self.fre_cyc

    def set_mu(self, mu: float):
        self.mu = float(mu)

    def reset(self, init_thetas=None, init_rs=None):
        if init_thetas is None:
            init_thetas = self.desired.copy()
        if init_rs is None:
            init_rs = np.ones(self.N, dtype=float) * math.sqrt(self.mu)
        self.theta = np.array(init_thetas, dtype=float) % (2.0 * math.pi)
        self.r = np.array(init_rs, dtype=float)

    def step(self):
        """
        Euler integration.
        With dt=0.002 this is usually stable for moderate alpha,k.
        """
        r = self.r
        th = self.theta

        # amplitude dynamics
        r_dot = self.alpha * (self.mu - r * r) * r

        # phase coupling dynamics
        # theta_dot_i = omega + sum_j K_ij * sin(theta_j - theta_i - phi_ij)
        # Build theta_j - theta_i matrix:
        th_i = th.reshape(-1, 1)     # [i,1]
        th_j = th.reshape(1, -1)     # [1,j]
        th_j_minus_th_i = th_j - th_i  # [i,j]

        coupling = np.sum(self.K * np.sin(th_j_minus_th_i - self.phi), axis=1)
        th_dot = self.omega + coupling

        # integrate
        self.r = r + r_dot * self.dt
        self.theta = (th + th_dot * self.dt) % (2.0 * math.pi)

    def get_phase(self, leg_id: int) -> float:
        return float(self.theta[int(leg_id)])

    def get_amp(self, leg_id: int) -> float:
        return float(self.r[int(leg_id)])


class LegPath(object):
    """
    使用 4 腿耦合振荡器网络生成足端理想椭圆轨迹：
      trg_x = x0 + a*cos(phi)
      trg_y = y0 + b*sin(phi)
    b 在 Up/Down 段切换（FU/FD、HU/HD）。
    """

    def __init__(self, fre_cyc: float, dt: float, desired_offsets=None, path_type: str = "ellipse", shape_params=None):
        super(LegPath, self).__init__()

        # 沿用你原来的中心与半轴
        self.para_FU = [[-0.00, -0.045], [0.03, 0.01]]
        self.para_FD = [[-0.00, -0.045], [0.03, 0.005]]
        self.para_HU = [[-0.005, -0.05], [0.03, 0.01]]
        self.para_HD = [[-0.005, -0.05], [0.03, 0.005]]

        if desired_offsets is None:
            # default trot
            desired_offsets = [0.0, math.pi, math.pi, 0.0]

        # 4-leg coupled CPG network
        # alpha: amplitude convergence speed
        # mu: target amplitude^2 (sqrt(mu) is steady r)
        # k: coupling strength (bigger -> faster lock-in, too big -> stiff)
        self.cpg = CPGNetwork4(
            fre_cyc=fre_cyc,
            dt=dt,
            desired_offsets=desired_offsets,
            alpha=30.0,
            mu=1.0,
            k=8.0,
        )
        self.path_type = str(path_type)
        self.shape_params = dict(shape_params or {})

    def reset(self, init_thetas=None, init_rs=None):
        self.cpg.reset(init_thetas=init_thetas, init_rs=init_rs)

    def set_shape_uniform(self, a: float, b: float):
        """
        Apply shared (a, b) to all four leg ellipse configs.
        """
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

    def step(self):
        """每个控制步推进一次：耦合振荡器网络积分"""
        self.cpg.step()

    # --------- CPG state getters (for logging / contact) ----------
    def get_leg_phase(self, leg_id: int) -> float:
        return self.cpg.get_phase(leg_id)

    def get_leg_amp(self, leg_id: int) -> float:
        return self.cpg.get_amp(leg_id)

    # --------- trajectory generator ----------
    def getCPGOvalPathPoint(self, leg_flag: str, leg_id: int, halfPeriod: float):
        """
        leg_flag: "F" / "H"
        leg_id: 0 FL, 1 FR, 2 HL, 3 HR
        halfPeriod: 你原来的 period（默认 1.0）
        """
        phi = self.cpg.get_phase(leg_id)  # [0, 2pi)

        # Up/Down 判据保持你原来的：phi < halfPeriod*pi
        is_up = (phi < float(halfPeriod) * math.pi)

        if leg_flag == "F":
            pathParameter = self.para_FU if is_up else self.para_FD
        else:
            pathParameter = self.para_HU if is_up else self.para_HD

        originPoint = pathParameter[0]  # [x0, y0]
        ovalRadius = pathParameter[1]   # [a, b]

        if self.path_type == "ellipse":
            trg_x = originPoint[0] + ovalRadius[0] * math.cos(phi)
            trg_y = originPoint[1] + ovalRadius[1] * math.sin(phi)
            return [trg_x, trg_y]

        trg_x, trg_y = self._new_path(phi, leg_flag, leg_id, halfPeriod, originPoint, ovalRadius, is_up)
        return [trg_x, trg_y]

    def _new_path(self, phi, leg_flag, leg_id, halfPeriod, originPoint, ovalRadius, is_up):
        """
        New path family chosen by data-driven ranking (Y1_quintic, Z1_quartic).
        Keep stance as legacy ellipse and only replace swing shaping.
        """
        a = float(ovalRadius[0])
        b = float(ovalRadius[1])
        x0 = float(originPoint[0])
        z0 = float(originPoint[1])

        if not is_up:
            return (
                x0 + a * math.cos(phi),
                z0 + b * math.sin(phi),
            )

        denom = max(float(halfPeriod) * math.pi, 1e-9)
        s = max(0.0, min(1.0, float(phi) / denom))
        p5 = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        y_shape = 1.0 - 2.0 * p5
        z_shape = 16.0 * s**2 * (1.0 - s)**2
        return (
            x0 + a * y_shape,
            z0 + b * z_shape,
        )
