import numpy as np
import time 
from collections import deque


class Motion_Planner:

    # List of np.array(2) trajecory points
    trajectory = deque([])
    traj_len = 0

    # np.array(2) point
    current_target = 0
    
    # prev velocity np.array(2)
    vel_B_p = 0

    # Float scalar value
    max_v = 0

    # Threshold for popping new target point
    threshold_d = 0

    def __init__(self, max_vel, trajectory=[], threshold = 0.002):
        # Initialize some parameters
        self.trajectory = deque(trajectory)
        self.max_v = max_vel
        if len(trajectory) != 0:
            self.current_target = self.trajectory.popleft()
            self.traj_len = len(self.trajectory)
        else:
            self.traj_len = -1
        
        self.threshold_d = threshold

    def update_trajectory_list(self, trajectory, override = False):
        # Override parameters allows scrapping of current trajectory map
        # Essentially dumps current trajecotry and adds new points
        # If override behaviour is false - then the passed trajectory points are appended

        if override:
            self.trajectory = deque(trajectory)
            self.current_target = self.trajectory.popleft()
        else:
            self.trajectory = self.trajectory + deque(trajectory)

        self.traj_len = len(self.trajectory)
    
    def compute_velocity(self, cur_pos):
        state = self.update_target(cur_pos)

        if not state:
            # return a zero velocity when nothing in trajectory
            return np.array([0,0])

        # Compute velocity and scale accordingly   
        print("Current target: {}".format(self.current_target))  
        delta_l = self.current_target - cur_pos
        delta_l_n = np.linalg.norm(self.current_target - cur_pos)
        # print("Normalized Velocity Before Scaling: {}".format(delta_l_n))
        # print("Allowed max velocity: {}".format(self.max_v))
        vel_B = 1/delta_l_n*min(delta_l_n,self.max_v)*delta_l
        print("Velocity: {}".format(vel_B))
        # print(delta_l.shape)
        # Update previous velocity
        self.vel_B_p = vel_B
        
        return vel_B

    def update_target(self, cur_pos):

        #
        delta_l = np.linalg.norm(self.current_target - cur_pos)
        print(delta_l)
        if delta_l <= self.threshold_d:
            if self.traj_len > 0:
                self.current_target = self.trajectory.popleft()
                self.traj_len = len(self.trajectory)
            else:
                print("No more traj points to follow")
                self.traj_len = -1
                return False
        
        return True

    def return_status(self):
        return self.traj_len
        





