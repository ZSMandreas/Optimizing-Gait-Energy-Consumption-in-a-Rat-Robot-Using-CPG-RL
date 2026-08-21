import numpy as np

class OscillatorLeg:
    def __init__(self, d_step=0.03, a=150.0):
        # 控制参数（可由RL控制）
        self.mu = 1.0       # 目标振幅
        self.omega = 2.0    # 振荡频率
        self.psi = 0.0      # 相位偏移速度

        # 振荡器状态变量
        self.r = 1.0         # 当前振幅
        self.r_dot = 0.0     # 振幅导数
        self.theta = 0.0     # 相位
        self.phi = 0.0       # 方向角

        # 参数
        self.a = a           # 收敛速度
        self.d_step = d_step # 步长缩放
        # self.h = np.random.uniform(0.035, 0.05)           # 身体高度
        # self.g_c = np.random.uniform(0.005, 0.02)       # 抬腿高度
        # self.g_p = np.random.uniform(0.002, 0.005)       # 支撑期高度
        self.h = 0.035
        self.g_c = 0.01
        self.g_p = 0.005
        
        
        

    def set_control(self, mu, omega, psi):
        self.mu = mu
        self.omega = omega
        self.psi = psi

    def step(self, dt):
        # 振幅二阶积分
        r_ddot = self.a * (self.a / 4 * (self.mu - self.r) - self.r_dot)
        self.r_dot += r_ddot * dt
        self.r += self.r_dot * dt

        # 相位积分
        self.theta += self.omega * dt
        self.phi += self.psi * dt

        # 保证在 0 ~ 2π 之间
        self.theta = self.theta % (2 * np.pi)
        self.phi = self.phi % (2 * np.pi)

    def get_foot_position(self):
        y = -self.d_step * (self.r - 1.0) * np.cos(self.theta)
        z = -self.h + (
            self.g_c * np.sin(self.theta) if np.sin(self.theta) > 0
            else self.g_p * np.sin(self.theta)
        )
        x = 0  # 或者固定的足部偏移（例如身体侧向偏移）

        return np.array([x, y, z])

    def get_joint_position(self,leg_M):
        foot_position = self.get_foot_position()
        trg_x, trg_y, trg_z = foot_position
        qVal = leg_M.pos_2_angle(trg_y,trg_z)
        return qVal
        

    def get_state(self):
        """返回当前振荡器状态: [r, ṙ, θ, θ̇, ϕ, ϕ̇]"""
        return np.array([
            self.r,                   # 振幅
            self.r_dot,               # 振幅导数
            self.theta,               # 相位
            self.omega,              # 相位速度
            self.phi,                # 方向角
            self.psi                 # 方向角速度
        ], dtype=np.float32)
