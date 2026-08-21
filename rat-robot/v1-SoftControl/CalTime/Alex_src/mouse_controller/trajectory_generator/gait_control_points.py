import numpy as np

# *******************************************************
# Type: Trajectory Generator Subscript
# 
# Computes the necessary control points to be passed to the trajectory
# generator class. Takes inputs from the gait parameters and velocity
# inputs from the high level controller. 
#
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

class Gait_Control_Parameters:

    # Standardized parameters
    # These will be adjusted during initilization with the actual 
    # gait parameters from our gait databank (determined by high-level)

    def __init__(self, gait_parameters):
        # Initialize the first set of parameters
        self.update_gait_parameters(gait_parameters)
        self.state_sign = [-1, 1]
        self.previous_point = np.array([self.neutral_stride_pos,self.neutral_stance_pos])
    
    def update_gait_parameters(self, gait_parameters):
        self.amp_swing = gait_parameters["amp_swing"]
        self.amp_stance = gait_parameters["amp_stance"]
        self.cycle_freq = gait_parameters["cycle_freq"]
        self.max_stride_length = gait_parameters["max_stride_length"]
        self.neutral_stance_pos = gait_parameters["neutral_stance_pos"]
        self.neutral_stride_pos = gait_parameters["neutral_stride_pos"]
        self.update_cycle_time(self.cycle_freq)
        self.amp_cycle = [self.amp_swing, -1*self.amp_stance]

    def update_cycle_time(self, cycle_freq):
        self.cycle_time = 1.0/cycle_freq

    def compute_control_points(self, vel: float, stance: int, alpha: float):
        # Computes the set of control points for the trajectory
        # INPUTS
        # velocity: integer     ||  value of the desired leg velocity for the next step
        # stance:   int         ||  boolean of whether leg is in swing (0) or stance (1)
        # alpha:    float       ||  velocity modulation factor for steering 0 < alpha < 1
        start_point = self.previous_point
        # print("Stance value: {}".format(stance))
        
        distance = np.abs(self.cycle_time * vel * alpha)
        stride = min(distance/15,3/2*self.max_stride_length)

        sign_m = np.sign(vel)
        # Gotten rid of if-clause statemenets here for control point generation
        mid_point = np.array([self.neutral_stride_pos,self.neutral_stance_pos + sign_m*self.amp_cycle[stance]])
        end_point = np.array([self.neutral_stride_pos + self.state_sign[stance]*stride, self.neutral_stance_pos])
       
        self.previous_point = end_point
        control_points = np.stack((start_point,mid_point,end_point))
        #print("Control points: {}".format(control_points))
        return control_points


