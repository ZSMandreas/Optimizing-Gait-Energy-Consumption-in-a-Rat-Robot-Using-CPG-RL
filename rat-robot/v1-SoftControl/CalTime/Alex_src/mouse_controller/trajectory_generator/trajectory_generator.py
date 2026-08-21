import numpy as np

# *******************************************************
# Type: Trajectory Generator
# 
# Combines a followable leg end-effector generator
# given a time, cycle time and leg velocity input
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

from mouse_controller.trajectory_generator.footend_generator import Footend_Traj_Generator
from mouse_controller.trajectory_generator.gait_control_points import Gait_Control_Parameters

class Leg_Trajectory_Generator:


    def __init__ (self, gait_parameters, leg_id):
        # Initialize the footend traj generator and control point generator

        self.footend_traj = Footend_Traj_Generator(1)
        self.cp_generator = Gait_Control_Parameters(gait_parameters)
        self.leg_id = leg_id
        self.set_traj_type()

    def set_traj_type(self, traj_type = 'cubic'):
        # Standard trajectory of cubic
        self.traj_type = traj_type

    def new_trajectory_compute(self, vel: float, stance: int, alpha: float): 
        # Compute a new trajectory (prompted by a state change)

        control_points = self.cp_generator.compute_control_points(vel,stance,alpha)
        self.current_trajectory = self.footend_traj.compute_trajectory(self.traj_type,
                                                                        control_points)
    
    def next_leg_point(self, current_cycle_time):
        # Compute the next goal point for the leg and return it

        return self.current_trajectory.forward_compute(current_cycle_time)
