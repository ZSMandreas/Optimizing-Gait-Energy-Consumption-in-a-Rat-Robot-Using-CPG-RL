# energy_utils.py
import numpy as np


class EnergyMeter:
    """
    简单机械能量/CoT 计量器。

    - 使用 τ * qdot 计算各电机的瞬时机械功率，只积累正功（电机做功部分）。
    - update(data, dt) 每次调用会：
        * 计算当前时间步的能量增量 E_inc
        * 更新总能量 self.E_total
        * 返回 (E_total, E_inc)
    - reset(data) 会把总能量清零，并记录当前前进方向的初始位置，用于之后计算 CoT。
    - cot(data) 使用当前总能量 + 初始/当前位移计算本 episode 的 CoT。
    """

    def __init__(self, model, motor_dof_idx, mass=None, forward_dof=1):
        """
        Args:
            model: mujoco.MjModel
            motor_dof_idx: list/array of dof 索引，对应你想计能量的所有电机
            mass: 机器人总质量，不给的话就用 model.body_mass 求和
            forward_dof: 前进方向在 qpos 中的坐标索引（一般 0:x, 1:y, 2:z）
                         你现在前进在 y 轴，所以默认 1。
        """
        self.model = model
        self.motor_dof_idx = np.array(motor_dof_idx, dtype=int)

        if mass is None:
            # 所有 body 质量求和作为整机质量
            self.mass = float(np.sum(model.body_mass))
        else:
            self.mass = float(mass)

        self.forward_dof = int(forward_dof)

        self.E_total = 0.0
        self._qpos_forward_0 = None  # 初始前进位置

    def reset(self, data=None):
        """
        清空能量，并在有 data 时记录当前前进方向位置作为 CoT 的起点。
        """
        self.E_total = 0.0
        if data is not None:
            self._qpos_forward_0 = float(data.qpos[self.forward_dof])
        else:
            self._qpos_forward_0 = None

    def update(self, data, dt):
        """
        用当前 data 更新能量。

        Args:
            data: mujoco.MjData
            dt:   仿真时间步长（例如 model.opt.timestep）

        Returns:
            (E_total, E_inc)
             E_total: 当前累计总能量
             E_inc:   本次调用对应的能量增量
        """
        tau = np.array(data.qfrc_actuator)[self.motor_dof_idx]   # 力矩
        qdot = np.array(data.qvel)[self.motor_dof_idx]           # 角速度

        P = tau * qdot                   # 各电机瞬时功率
        P_pos = np.maximum(P, 0.0)       # 只算正功
        P_sum = float(np.sum(P_pos))     # 所有电机功率求和

        E_inc = P_sum * dt               # 本时间步能量增量
        self.E_total += E_inc

        return self.E_total, E_inc

    def distance(self, data):
        """
        计算从 reset 时刻到当前的前进方向位移（绝对值）。
        """
        if self._qpos_forward_0 is None:
            return 0.0
        y_now = float(data.qpos[self.forward_dof])
        return abs(y_now - self._qpos_forward_0)

    def cot(self, data, g=9.81):
        """
        计算当前 episode 的 CoT = E_total / (m g D)。

        如果 D 太小（几乎没动），返回 inf。
        """
        D = self.distance(data)
        if D <= 1e-6:
            return float("inf")
        return float(self.E_total / (self.mass * g * D))
