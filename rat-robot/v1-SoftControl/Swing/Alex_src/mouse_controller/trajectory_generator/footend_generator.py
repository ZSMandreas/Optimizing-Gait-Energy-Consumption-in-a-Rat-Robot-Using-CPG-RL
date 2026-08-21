import numpy as np
import time 
from collections import deque
import math
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R

# *******************************************************
# Type: Trajectory Generator Subscript
# 
# Trajectory generator for leg endeffectors
# based on a set of control points and a desired cycle time
# The trajectories are stored as custom functions that can
# be sampled during the locomotion cycle and used to plan
# the next points for the leg.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

class Footend_Traj_Generator(object):

    def __init__(self, type):
        # Initialize our trajectory generator class
        self.trajectory_type = type
        self.parabolic_factors = np.zeros((6,2))
        self.cubic_factors = np.zeros((8,2))

    def compute_trajectory(self, type, control_points, cycle_time=1):
        # Top level method for computing trajectories
        if type == 'linear':
            return self.linear_trajectory(control_points, cycle_time)
        elif type == 'parabolic':
            return self.parabolic_trajectory(control_points, cycle_time)
        elif type == 'cubic':
            return self.cubic_trajectory(control_points, cycle_time)
        else:
            return "Invalid trajectory type passed"
        

    def linear_trajectory(self, control_points, cycle_time):
        # Computes curve coefficient for linear trajectory 
        # Only supports 3 control points
        # Inputs:
        # control_points: Numpy matrix of trajectory points (3x2)
        t = 2/cycle_time
        trajector_factor_matrix = np.array([[-t, t, 0],
                                            [1, 0, 0],
                                            [0, -t, t],
                                            [0, 1, 0]])
        trajectory_factors = np.dot(trajector_factor_matrix,control_points)
        linear_trajectory = Linear_Trajectory(control_points, trajectory_factors, cycle_time)
        return linear_trajectory
    
    def parabolic_trajectory(self, control_points, cycle_time):
        # Compute curve cofficient for parabolic trajectory given 3 control points
        # Only supports 3 control points
        # Inputs:
        # control_points: Numpy matrix of trajectory points (3x2)
        t = 2/cycle_time
        trajectory_factor_matrix_y = np.array([[t**2, -3*t**2, 2*t**2],
                                                [-2*t, 4*t, -2*t],
                                                [1, 0, 0],
                                                [0, t**2, -t**2],
                                                [0, -2*t, 2*t],
                                                [0, 1, 0]])
        # The below matrix is a hybrid parabolic-linear, where y is linear and z parabolic
        # Need to conduct testing to see how well this will work
        trajectory_factor_matrix_y_lin = np.array([[0 , 0 , 0],
                                                [-t, t, 0],
                                                [1, 0, 0],
                                                [0, 0, 0],
                                                [0, -t, t],
                                                [0, 1, 0]])
        
        self.parabolic_factors[:,0] = np.dot(trajectory_factor_matrix_y,control_points[:,0])
        trajectory_factor_matrix_z = np.array([[1/2*t**2, -t**2, 1/2*t**2],
                                                [-3/2*t, 2*t, -1/2*t],
                                                [1, 0, 0],
                                                [1/2*t**2, -t**2, 1/2*t**2],
                                                [-1/2*t, 0, 1/2*t],
                                                [0, 1, 0]])
        self.parabolic_factors[:,1] = np.dot(trajectory_factor_matrix_z,control_points[:,1])
        parabolic_trajectory = Parabolic_Trajectory(control_points, self.parabolic_factors, cycle_time)
        return parabolic_trajectory

    def cubic_trajectory(self, control_points, cycle_time):
        # Compute curve coefficient for cubic trajectory given 3 control points
        # Only supports 3 control points
        # Inputs:
        # control_points: Numpy matrix of trajectory points (3x2)
        t = 2/cycle_time
        trajectory_factor_matrix_y = np.array([[5/4*t**3, -2*t**3, 3/4*t**3],
                                                [-9/4*t**2, 3*t**2, -3/4*t**2],
                                                [0, 0, 0],
                                                [1, 0, 0],
                                                [-3/4*t**3, 2*t**3, -5/4*t**3],
                                                [3/2*t**2, -3*t**2, 3/2*t**2],
                                                [-3/4*t, 0, 3/4*t],
                                                [0, 1, 0]])
        self.cubic_factors[:,0] = np.dot(trajectory_factor_matrix_y,control_points[:,0])
        trajectory_factor_matrix_z = np.array([[1/4*t**3, -1/2*t**3, 1/4*t**3],
                                                [0, 0, 0],
                                                [-5/4*t, 3/2*t, -1/4*t],
                                                [1, 0, 0],
                                                [-1/4*t**3, 1/2*t**3, -1/4*t**3],
                                                [3/4*t**2, -3/2*t**2, 3/4*t**2],
                                                [-1/2*t, 0, 1/2*t],
                                                [0, 1, 0]])
        self.cubic_factors[:,1] = np.dot(trajectory_factor_matrix_z,control_points[:,1])
        cubic_trajectory = Cubic_Trajectory(control_points, self.cubic_factors, cycle_time)
        return cubic_trajectory

class Linear_Trajectory(object):
    # Class that contains equation variables for a linear trajectory
    # Important quantities
    # Control_points: matrix of (3x2) of the trajectory points
    # trajector_factors: matrix of (4x2) of equation variables a1, b1, a2, b2
    # cycle_time: time taken for the full trajectory

    control_points = np.zeros((3,2))
    trajectory_factors = np.zeros((4,2))
    cycle_time = 0

    def __init__(self, control_points, trajectory_factors, cycle_time):
        # Initializing a linear trajectory module
        self.control_points = control_points
        self.trajectory_factors = trajectory_factors
        self.cycle_time = cycle_time 

    def forward_compute(self, current_cycle_time):
        # Compute the desired current value point forward in the trajectory
        if (current_cycle_time < self.cycle_time/2):
            # compute using first segment
            return self.linear_first_segment_compute(current_cycle_time)
        elif (current_cycle_time > self.cycle_time):
            # return the end control point here
            return self.control_points[2]
        else:
            # compute using second segment
            return self.linear_second_segment_compute(current_cycle_time)

    def linear_first_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        next_point = self.trajectory_factors[0]*current_cycle_time + self.trajectory_factors[1]
        return next_point

    def linear_second_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        next_point = self.trajectory_factors[2]*(current_cycle_time-self.cycle_time/2) + self.trajectory_factors[3]
        return next_point

class Parabolic_Trajectory(object):
    # Class that contains equation variables for a linear trajectory
    # Important quantities
    # Control_points: matrix of (3x2) of the trajectory points
    # trajector_factors: matrix of (6x2) of equation variables a1, b1, c1, a2, b2, c2
    # cycle_time: time taken for the full trajectory

    control_points = np.zeros((3,2))
    trajectory_factors = np.zeros((6,2))
    cycle_time = 0

    def __init__(self, control_points, trajectory_factors, cycle_time):
        # Initializing a linear trajectory module
        self.control_points = control_points
        self.trajectory_factors = trajectory_factors
        self.cycle_time = cycle_time 

    def forward_compute(self, current_cycle_time):
        # Compute the desired current value point forward in the trajectory
        if (current_cycle_time < self.cycle_time/2):
            # compute using first segment
            return self.parabolic_first_segment_compute(current_cycle_time)
        elif (current_cycle_time > self.cycle_time):
            # return the end control point here
            return self.control_points[2]
        else:
            # compute using second segment
            return self.parabolic_second_segment_compute(current_cycle_time)

    def parabolic_first_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        a1 = self.trajectory_factors[0]
        b1 = self.trajectory_factors[1]
        c1 = self.trajectory_factors[2]
        next_point = a1*current_cycle_time**2 + b1*current_cycle_time + c1
        return next_point

    def parabolic_second_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        a2 = self.trajectory_factors[3]
        b2 = self.trajectory_factors[4]
        c2 = self.trajectory_factors[5]
        h_t = current_cycle_time - self.cycle_time/2
        next_point = a2*(h_t)**2 + b2*(h_t) + c2
        return next_point

class Cubic_Trajectory(object):
    # Class that contains equation variables for a linear trajectory
    # Important quantities
    # Control_points: matrix of (3x2) of the trajectory points
    # trajector_factors: matrix of (8x2) of equation variables a1, b1, c1, d1, a2, b2, c2, d2
    # cycle_time: time taken for the full trajectory

    control_points = np.zeros((3,2))
    trajectory_factors = np.zeros((8,2))
    cycle_time = 0

    def __init__(self, control_points, trajectory_factors, cycle_time):
        # Initializing a linear trajectory module
        self.control_points = control_points
        self.trajectory_factors = trajectory_factors
        self.cycle_time = cycle_time 

    def forward_compute(self, current_cycle_time):
        # Compute the desired current value point forward in the trajectory
        if (current_cycle_time < self.cycle_time/2):
            # compute using first segment
            return self.cubic_first_segment_compute(current_cycle_time)
        elif (current_cycle_time > self.cycle_time):
            # return the end control point here
            return self.control_points[2]
        else:
            # compute using second segment
            return self.cubic_second_segment_compute(current_cycle_time)

    def cubic_first_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        a1 = self.trajectory_factors[0]
        b1 = self.trajectory_factors[1]
        c1 = self.trajectory_factors[2]
        d1 = self.trajectory_factors[3]
        t = current_cycle_time
        next_point = a1*t**3 + b1*t**2 + c1*t + d1
        return next_point

    def cubic_second_segment_compute(self, current_cycle_time):
        # Compute the points for the first segment
        # Return array (1,2)
        a2 = self.trajectory_factors[4]
        b2 = self.trajectory_factors[5]
        c2 = self.trajectory_factors[6]
        d2 = self.trajectory_factors[7]
        h_t = current_cycle_time - self.cycle_time/2
        next_point = a2*(h_t)**3 + b2*(h_t)**2 + c2*(h_t) + d2
        return next_point
