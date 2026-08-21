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
# Import the leg and motionplanner modules
from mouse_controller.leg_unit_class import Leg_Unit

# ************** Four Legged Walker Class *******************
# asd
# 
# 
# 
# ***********************************************************


class Quadruped_Walker:
    
    body_rot = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    body_pos = 0

    body_pos_dot = 0
    body_rot_dot = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

    legs = 4

    # Order [fl, fr, rl, rr]
    legs_lifted = np.array([0,0,0,0])

    def __init__(self):
        self.new_mouse()

    def new_mouse(self):
        self.max_vel = 0.1
        fr_t1_param = {'lr0':0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.025, 
                    'l2': 0.018, 'l3': 0.028, 'l4': 0.015,'theta3':np.pi/10}

        self.lu_fl = Leg_Unit('fr3',fr_t1_param,self.max_vel,threshold=0.002)
        self.lu_fr = Leg_Unit('fr3',fr_t1_param,self.max_vel,threshold=0.002)

        # Define the necessary leg parameters for rear legs
        rr_t3_param = {'lr0':0.02678, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.03, 
                        'l2': 0.0084, 'l3': 0.045, 'l4': 0.025,'theta3':110*np.pi/180}

        self.lu_rl = Leg_Unit('rr3',rr_t3_param,self.max_vel,threshold=0.002)
        self.lu_rr = Leg_Unit('rr3',rr_t3_param,self.max_vel,threshold=0.002)

        self.mouse = {"fl": self.lu_fl,
                    "fr": self.lu_fr,
                    "rl": self.lu_rl,
                    "rr": self.lu_rr}
    
    def top_level_kinematics(self,vel,rot,current_servos,time_step=0.1):
        # Method that returns the leg positions for moving the body


        leg_velocities = self.computing_leg_velocities(vel,rot)
        q_legs = self.passing_velocities_to_legs(leg_velocities, current_servos, time_step)

        # Combine leg q-vals and the spine q_val together
        # Spine q_val taken via horizontal velocity space
        q_vals = np.concatenate((q_legs,vel[0]))

        return q_vals
    
    def trajectory_level_kinematics(self,trajectory,current_servos,time_step=0.1):
        # 
        self.mouse["fl"].new_trajectory(trajectory)
        self.mouse["fr"].new_trajectory(trajectory)
        self.mouse["rl"].new_trajectory(trajectory)
        self.mouse["rr"].new_trajectory(trajectory)
        q_legs = self.trajectory_q_vals(current_servos, time_step)

        q_vals = np.concatenate((q_legs,np.array([0])))

        return q_vals
    
    def trajectory_q_vals(self, current_servos,time_step):
        # Compute the values for the servos to move to trajectory points

        csv_fl = current_servos[:]
        csv_fr = current_servos[4:]
        csv_rl = current_servos[8:]
        csv_rr = current_servos[12:]

        q_fl = self.mouse["fl"].kinematic_update(csv_fl,time_step)
        q_fr = self.mouse["fr"].kinematic_update(csv_fr,time_step)
        q_rl = self.mouse["rl"].kinematic_update(csv_rl,time_step)
        q_rr = self.mouse["rr"].kinematic_update(csv_rr,time_step)

        return np.concatenate((q_fl,q_fr,q_rl,q_rr))


    def passing_velocities_to_legs(self,leg_velocities,current_servos,time_step):
        # Method to pass velocities onto legs for processing
        # 
        vel_fl = leg_velocities[1:,0]
        vel_fr = leg_velocities[1:,1]
        vel_rl = leg_velocities[1:,2]
        vel_rr = leg_velocities[1:,3]

        csv_fl = current_servos[:]
        csv_fr = current_servos[4:]
        csv_rl = current_servos[8:]
        csv_rr = current_servos[12:]

        q_fl = self.mouse["fl"].kinematic_update_no_mp(vel_fl,csv_fl,time_step,self.legs_lifted[0])
        q_fr = self.mouse["fr"].kinematic_update_no_mp(vel_fr,csv_fr,time_step,self.legs_lifted[1])
        q_rl = self.mouse["rl"].kinematic_update_no_mp(vel_rl,csv_rl,time_step,self.legs_lifted[2])
        q_rr = self.mouse["rr"].kinematic_update_no_mp(vel_rr,csv_rr,time_step,self.legs_lifted[3])

        return np.concatenate((q_fl,q_fr,q_rl,q_rr))

    def computing_leg_velocities(self,vel,rot):
        # Method for computing the inverse leg velocities
        # 
        # 
        # 
        # OUTPUT: Leg velocity vector (3,4) 

        grounded_legs = self.legs - self.legs_lifted.sum()

        vel_fpg = np.zeros((3,self.legs))
        vel_xb = np.tile(vel,self.legs)
        pos_fpl = self.combined_leg_positions().transpose()

        mid_val = vel_fpg - vel_xb - np.dot(rot,pos_fpl)
        leg_velocities = np.dot(self.body_rot.transpose(), mid_val)

        return leg_velocities

    def combined_leg_positions(self):
        # Method that returns the (3,4) leg position matrix
        # 
        # 
        # 
        # OUTPUT: Leg position vector (3,4) 

        leg_fl = self.mouse["fl"].current_leg_position()
        leg_fr = self.mouse["fr"].current_leg_position()
        leg_rl = self.mouse["rl"].current_leg_position()
        leg_rr = self.mouse["rr"].current_leg_position()

        return np.array([leg_fl, leg_fr, leg_rl, leg_rr])






