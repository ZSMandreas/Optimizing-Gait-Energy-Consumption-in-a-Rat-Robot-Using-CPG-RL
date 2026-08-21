import numpy as np
from GO2state import State

class StateProcessor:
    def __init__(self, obs_size, n=5):
        self.n = n
        self.obs_buffer = []  # 存储最近 n 步观测
        self.next_obs_buffer = []
    def get_obs_buffer(self):
        return self.get_obs_buffer
    
    def get_next_obs_buffer(self):
        return self.get_next_obs_buffer
    
    def update_observations(self, obs):
        # 更新历史观测值
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > self.n:
            self.obs_buffer.pop(0)

    def get_observation(self):
        pos_ms = []
        vel_ms = [] 
        pos_ms.append(State.get_xpos())
        vel_ms.append(State.get_xvel())
        return np.concatenate(pos_ms,vel_ms) 
    
    def update_next_observations(self, next_obs):
        self.next_obs_buffer.append(next_obs)
        if len(self.next_obs_buffer) > self.n:
            self.next_obs_buffer.pop(0)

    def get_state(self, current_action):
        # 计算前 n 步和后 n 步观测值的均值
        mean_obs_prev = np.mean(self.obs_buffer, axis=0) if self.obs_buffer else np.zeros_like(current_action)
        mean_obs_next = np.mean(self.next_obs_buffer, axis=0) if self.next_obs_buffer else np.zeros_like(current_action)
        obs_diff = mean_obs_prev - mean_obs_next  # 差值代表趋势

        # 拼接新的状态
        state = np.concatenate([mean_obs_prev, current_action, mean_obs_next, obs_diff])
        return state
    
    
    def unscale_action(self, action):
        """
        将 [-1, 1] 范围的动作映射到电机的实际控制范围
        """
        low = np.array([m[0] for m in self.motor_ranges])
        high = np.array([m[1] for m in self.motor_ranges])
        return low + (action + 1.0) * 0.5 * (high - low)

    def scale_action(self, unscaled_action):
        """
        将电机的实际控制范围动作反归一化回 [-1, 1]
        """
        low = np.array([m[0] for m in self.motor_ranges])
        high = np.array([m[1] for m in self.motor_ranges])
        return 2.0 * (unscaled_action - low) / (high - low) - 1.0   
