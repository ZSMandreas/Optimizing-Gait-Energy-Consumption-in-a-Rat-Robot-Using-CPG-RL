#!/usr/bin/env python3

import os
import math
import numpy as np
import random
import pandas as pd
import time
import pathlib

from mouse_controller.leg_models.front_leg_t1 import Front_Leg1  
from mouse_controller.leg_models.front_leg_t3 import Front_Leg3
from mouse_controller.leg_models.rear_leg_t1 import Rear_Leg1
from mouse_controller.leg_models.rear_leg_t3 import Rear_Leg3

from mouse_controller.motion.motion_planner import Motion_Planner

# *******************************************************
# Type: Leg Unit Class
# 
# Combines the leg model and motion planner for the legs
# Allows for a more compact representation
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 04.02.2021
# *********************************************************

class Leg_Unit:

    time_step = 0.001
    leg_model = 0
    mp_leg = 0

    leg_models = {'fr1' : Front_Leg1,
                  'fr3' : Front_Leg3,
                  'rr1' : Rear_Leg1,
                  'rr3' : Rear_Leg3}

    def __init__(self, leg_type, leg_param, max_vel=0.1, threshold=0.0005) -> None:
        self.time_step = 0.001
        self.leg_model = self.leg_models[leg_type](leg_param)
        self.mp_leg = Motion_Planner(max_vel = max_vel, threshold = threshold)
        print("New leg initialized")

    def add_to_trajectory(self, trajectory) -> None:
        # This methods adds additional trajectory points to an existing trajectory
        # INPUT: 
        #       trajectory: List of (2,) numpy arrays 
        # OUTPUT: 
        #       none
        self.mp_leg.update_trajectory_list(trajectory,override=False)

    def new_trajectory(self, trajectory) -> None:
        # This methods dumps the old trajectory and adds the new trajectory
        # INPUT: 
        #       trajectory: List of (2,) numpy arrays 
        # OUTPUT: 
        #       none
        self.mp_leg.update_trajectory_list(trajectory,override=True)

    def mp_status(self) -> int:
        # This method accesses the current trajectory length of the mp
        # The value is returned as a status
        # Status = -1 means the motion planner is empty
        # INPUT: 
        #       none
        # OUTPUT: 
        #       status: int -1 or any positive number 

        status = self.mp_leg.return_status()

        print("Status of MP: {}".format(status))
        return status

    def current_leg_position(self):
        yH = self.leg_model.leg_param['yH']
        zH = self.leg_model.leg_param['zH']
        
        return np.array([0,yH, zH])

    def current_leg_servos(self):
        q1 = self.leg_model.leg_param['q1']
        q2 = self.leg_model.leg_param['q2']
        
        return np.array([q1,q2])


    def kinematic_update(self, current_servos, time_step):
        # This methods runs through the kinematic update steps and returns the new
        # servo values
        # INPUT: 
        #       current_servos: (2,) numpy array || servo values from simulation
        #       time_step:      float            || integration timestep to use
        # OUTPUT: 
        #       q_vals: (2,) numpy array         || new control servo values
        start_time = time.perf_counter()
        self.leg_model.update_servo_pos(current_servos)


        # Then run the forward kinematic in the leg with the current servo values
        cur_pos = self.leg_model.forward_kinematics()
        print("Current End-Effector Position: {}".format(cur_pos))
        # Then compute the needed velocity via the motion planner and current leg leg position
        vel_B = self.mp_leg.compute_velocity(cur_pos)
        print("Velocity of endpoint: {}".format(np.linalg.norm(vel_B)))
        if self.mp_status() != -1:
            len_vel = len(time_step)
            time_step_j = time_step[len_vel-self.mp_status()-1]
        else:
            time_step_j = self.time_step

        # Lastly evaluate the new servo values and return them
        q_vals = self.leg_model.inverse_kinematics(vel_B,time_step_j)

        end_time = time.perf_counter()
        print("Time taken for update cycle: {}".format(end_time-start_time))

        return q_vals

    def kinematic_update_no_mp(self, leg_vel, current_servos, time_step):

        # Update the leg models with current servo positions
        self.leg_model.update_servo_pos(current_servos)


        # Then run the forward kinematic in the leg with the current servo values
        self.leg_model.forward_kinematics()
        # Then compute the needed velocity via the motion planner and current leg leg position

        # Lastly evaluate the new servo values and return them
        q_vals = self.leg_model.inverse_kinematics(leg_vel,time_step)

        return q_vals