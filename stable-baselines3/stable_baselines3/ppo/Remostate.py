from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from CPGcontroller import OscillatorLeg

class State:
    
    def __init__(self, model, data, n):
        self.model = model
        self.data = data
        self.n = n
        self.obs_buffer = []  
        self.next_obs_buffer = []
        self.obs_buffer_2 = []
        self.legRealPoint_x = [[],[],[],[]]
        self.legRealPoint_y = [[],[],[],[]] 
        self.f_condition = []  
        self.real_trajectory = []
        self.n_legs = 4
        self.cpgs = [OscillatorLeg() for _ in range(self.n_legs)]
        cpg_buffer = [[] for _ in range(self.n_legs)]
    def get_body_position(self):
        # print(f"body_position{self.data.sensordata[16:19]}")
        return self.data.sensordata[16:19] * 1
    
    def get_body_velocity(self):
        # print(f"body_position{self.data.sensordata[23:26]}")
        return self.data.sensordata[23:26] * 1
    
    def get_body_angular_velocity(self):
        return self.data.sensordata[29:] * 1
    
    def get_body_rotation_euler(self):
        # 确保四元数顺序为 [w, x, y, z]
        quat = self.data.sensordata[19:23] * 1
        # quat = [quat[1], quat[2], quat[3], quat[0]]
        # euler = Rotation.from_quat(quat).as_euler('xyz')  # 删除 scalar_first
        return quat
    
    def get_feet_condition(self):
        feet_pos = self.data.sensordata[12:16]
        for i in range(4):
            if feet_pos[i] > 0:
                feet_pos[i] = 1
            else:
                feet_pos[i] = 0
        # print(f"feet_pos{feet_pos}")
        return feet_pos
    
    
    def time(self):
        return self.data.time * 1
    
    def update_time(self, dt):
        self.data.time += dt
    
    
    def get_data(self):
        return self.data
    
    def update_state(self, new_data):
        self.data = new_data
        
    def get_joints_pos(self):
        # print(self.data.sensordata)
        return self.data.sensordata[0:8] * 1
    
    def get_other_angle(self):
        return self.data.sensordata[8:12] * 1
    
    def get_IMU_acc(self):
        return self.data.sensordata[26:29] * 1

    
    # def get_xpos(self):
    #     joints_pos = self.get_joints_pos()
    #     other_pos = self.get_other_angle()
    #     feet_condition = self.get_feet_condition()
    #     base_pos = self.get_body_position()
    #     base_rot = self.get_body_rotation_euler()
    #     return np.hstack([joints_pos, other_pos,feet_condition, base_pos, base_rot]) 

    def get_xpos(self):
        # joints_pos = self.get_joints_pos()
        # other_pos = self.get_other_angle()
        # feet_condition = self.get_feet_condition()
        base_pos = self.get_body_position()
        base_rot = self.get_body_rotation_euler()
        return np.hstack([base_pos, base_rot]) 
    
    def get_xvel(self):
        base_vel = self.get_body_velocity()
        base_acc = self.get_IMU_acc()
        base_omega = self.get_body_angular_velocity()
        return np.hstack([base_vel, base_acc, base_omega]) 

    def get_obs_buffer(self):
        return self.obs_buffer
    
    def get_next_obs_buffer(self):
        return self.next_obs_buffer
    
    def get_obs_buffer_2(self):
        return self.obs_buffer_2
    
    def update_observations(self, obs):
        # 更新历史观测值
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > self.n:
            self.obs_buffer.pop(0)

    def get_observation(self):
        pos_ms = self.get_xpos()
        vel_ms = self.get_xvel()
        # print(pos_ms)
        # print(vel_ms)
        # print(self.data.sensordata)
        return np.hstack([pos_ms,vel_ms]) 
    
    def update_next_observations(self, next_obs):
        self.next_obs_buffer.append(next_obs)
        if len(self.next_obs_buffer) > self.n:
            self.next_obs_buffer.pop(0)
            
    def reset_observation(self):
        self.obs_buffer.clear()
        # print(f"obs_buffer{self.obs_buffer}")
       
    
    def reset_next_observation(self):
        self.next_obs_buffer = []
     
    def foot_trajectory(self):
        self.legPosName = [
			["thigh_link_fl", "ankle_fl"],
			["thigh_link_fr", "ankle_fr"],
			["thigh_link_rl", "ankle_rl"],
			["thigh_link_rr", "ankle_rr"]]            
        for i in range(4):
            origin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.legPosName[i][0])
            current_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.legPosName[i][1])      
            originPoint = self.data.site_xpos[origin_id]
            currentPoint = self.data.site_xpos[current_id]

            #print(originPoint, currentPoint)
            tX = currentPoint[1]-originPoint[1]
            tY = currentPoint[2]-originPoint[2]
            self.legRealPoint_x[i].append(tX)
            self.legRealPoint_y[i].append(tY)
    def initializing(self):
        self.legRealPoint_x = [[],[],[],[]]
        self.legRealPoint_y = [[],[],[],[]]   
    def save_foot_trajectory(self):
        # 保存为 npz 文件（压缩打包多个数组）
        # np.savez("leg_real_traj_with_spine.npz",
        #  leg_x=np.array(self.legRealPoint_x),  # shape: (4, T)
        #  leg_y=np.array(self.legRealPoint_y))  # shape: (4, T)
        np.savez("leg_real_traj_no_spine.npz",
         leg_x=np.array(self.legRealPoint_x),  # shape: (4, T)
         leg_y=np.array(self.legRealPoint_y))  # shape: (4, T)
        
    def get_real_trajectory(self):
        base_position = self.get_body_position()
        base_x = base_position[0]
        base_y = base_position[1]
        self.real_trajectory.append((base_x,base_y))
        
    def save_real_trajectory(self):
        np.save("real_trajectory_no_spine.npy",self.real_trajectory)
        # np.save("real_trajectory_with_spine",self.real_trajectory)
            
    # cpg_state = np.concatenate([c.get_state() for c in self.cpgs])
    def get_state(self, current_action):
        # 计算前 n 步和后 n 步观测值的均值
        mean_obs_prev = np.mean(self.obs_buffer, axis=0) if self.obs_buffer else np.zeros_like(current_action)
        mean_obs_next = np.mean(self.next_obs_buffer, axis=0) if self.next_obs_buffer else np.zeros_like(current_action)

        # 拼接新的状态
        state = np.concatenate([mean_obs_prev, current_action, mean_obs_next])
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






    
        
        
    
        