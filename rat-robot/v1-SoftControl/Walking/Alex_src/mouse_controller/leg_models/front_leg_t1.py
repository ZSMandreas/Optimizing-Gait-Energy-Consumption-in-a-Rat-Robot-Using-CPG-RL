import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
import time

# *******************************************************
# Type: Model Script
# 
# Model of the front leg type 1. 
# The mathematical foundations are described via the 
# respective thesis sections. Type 1 legs model tendon wrapping
# behaviour induced by the leg position
# This class stores a digital twin of the rear leg with all
# relevant leg parameters. 
# This model contains functions for computing the forward
# and inverse kinematics of the legs.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 04.02.2021
# *********************************************************

class Front_Leg1:

    # Dictionary contains all relevant leg parameters for kinematic modelling
    # Constants: l1, l2, rp, d1, lr0 [initialized with]
    # Motor inputs: q1, q2
    # Velocities: q1_dot, q2_dot
    # Modelling variables: theta1, theta2
    # End point: yB, zB
    # Target end_point: yB_t, zB_t
    # Also contains all parameters of previous position (k-1) with subscript *_p
    leg_param = {'q1': 0, 'q2': 0,
                'q1_p': 0, 'q2_p': 0,
                'q1_dot':0, 'q2_dot':0,
                'q1_dot_p': 0, 'q2_dot_p': 0,
                'yB': 0, 'zB': 0,
                'yB_p': 0, 'zB_p': 0,
                'yB_dot': 0, 'zB_dot': 0,
                'yB_dot_p': 0, 'zB_dot_p': 0,
                'yH': 0, 'zH': 0,
                'yH_p': 0, 'zH_p': 0,
                'yH_dot': 0, 'zH_dot': 0,
                'yH_dot_p': 0, 'zH_dot_p': 0,
                'theta1': 0, 'theta2': 0, 'theta3': np.pi/10,
                'theta1_p': 0, 'theta2_p': 0, 'theta3_p': np.pi/10,
                'theta1_dot':0, 'theta2_dot':0, 'theta3_dot': 0,
                'theta1_dot_p':0, 'theta2_dot_p':0, 'theta3_dot_p': 0} 
    # Currently theta2_dot / theta2_dot_p not being updated/considered

    constraints = {'q1_min': -np.pi/2, 'q1_max': np.pi/2,
                    'q2_min': -3/2*np.pi, 'q2_max': np.pi/2}


    def __init__(self, leg_params):
        for i, value in enumerate(leg_params):
            self.leg_param[value] = leg_params[value]

        print("Constant import complete.")
        # Right after initialization compute all necessary modelling values
        # Default position q1,q2 = (0,0)
        self.forward_kinematics()
        print("First pass FK complete.")

    def update_servo_pos(self, pos):
        # This method feeds back the measured real servo values into the model
        # INPUT: 
        #       pos: (2,) numpy vector   || current measured servo values
        # OUTPUT: 
        #       none

        print("Current q1: {} || Current q2: {}".format(self.leg_param['q1'],self.leg_param['q2']))
        self.leg_param['q1'] = pos[0]
        self.leg_param['q2'] = pos[1]
        print("Current q1: {} || Current q2: {}".format(self.leg_param['q1'],self.leg_param['q2']))
        # self.leg_param['q1'] = (self.leg_param['q1'] + pos[0])/2
        # self.leg_param['q2'] = (self.leg_param['q2'] + pos[1])/2

    def perform_kinematic_update(self, vel_B, time_step = 0.02):
        # Performs a full motion update for the front leg
        # Should be used during trajectory following 
        # Does a single forward kinematics pass
        # Then computes the next set of servo angles to follow velocity
        # INPUT: vel_B - array of size (2,) - velocity of end point as needed by trajectory
        # RETURN: Array of size (2,) - (q1, q2) 

        self.inverse_kinematics(vel_B, time_step)
        self.forward_kinematics()
        # Return the new q1,q2 parameters to be controlled
        return [self.leg_param['q1'], self.leg_param['q2']]

    # ******************* FORWARD KINEMATIC ***********************       
    def forward_kinematics(self):
        # This methods run through the forward kinematic chain
        # INPUT: none
        # OUTPUT: 
        #       H_pos: (2,) numpy vector || leg end-effector position

        # Compute current values of the modelling variables
        self.compute_theta_2()
        self.compute_theta_1()

        # Run the forward kinematics chain here
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        l3 = self.leg_param['l3']
        l4 = self.leg_param['l4']
        q1 = self.leg_param['q1']
        theta1 = self.leg_param['theta1']
        theta3 = self.leg_param['theta3']
        
        yB = l1*np.sin(q1) - l2*np.sin(theta1 - q1)
        zB = -l1*np.cos(q1) - l2*np.cos(theta1 - q1)

        # Full leg forward kinematics
        yH = l1*np.sin(q1) - l3*np.sin(theta1-q1) - l4*np.sin(theta3 + theta1 -q1)
        zH = -l1*np.cos(q1) - l3*np.cos(theta1-q1) - l4*np.cos(theta3 + theta1 -q1)

        self.leg_param['yB_p'] = self.leg_param['yB']
        self.leg_param['zB_p'] = self.leg_param['zB']
        self.leg_param['yB'] = yB
        self.leg_param['zB'] = zB
        self.leg_param['yH_p'] = self.leg_param['yH']
        self.leg_param['zH_p'] = self.leg_param['zH']
        self.leg_param['yH'] = yH
        self.leg_param['zH'] = zH

        return np.array([self.leg_param['yH'],self.leg_param['zH']])
        
    def compute_theta_2(self):
        # Compute theta2 angle from q1 and q2 of servos
        # INPUT: none
        # OUTPUT: none

        # Unload necessary equation parameters
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        rp = self.leg_param['rp']
        d1 = self.leg_param['d1']
        lr0 = self.leg_param['lr0']
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']

        # Numeri equation stemming from Matlab solver
        theta2_fnc = lambda x : (  (np.cos(x - np.arctan((lr0 + q2*rp + rp*x)/rp))*np.sqrt((lr0 + q2*rp + rp*x)**2 + rp**2) + l1*np.sin(q1))**2 
                                 + (d1 + np.sin(x - np.arctan((lr0 + q2*rp + rp*x)/rp))*np.sqrt((lr0 + q2*rp + rp*x)**2 + rp**2) + l1*np.cos(q1))**2 
                                 - l2**2 )

        sol_theta2 = opt.fsolve(theta2_fnc, 1)

        self.leg_param['theta2_p'] = self.leg_param['theta2']
        self.leg_param['theta2'] = sol_theta2[0]

    def compute_theta_1(self):
        # Compute theta1 angle for the leg after solving for theta2
        # INPUT: none
        # OUTPUT: none

        # Unload necessary equation parameters
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        rp = self.leg_param['rp']
        d1 = self.leg_param['d1']
        lr0 = self.leg_param['lr0']
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']
        theta2 = self.leg_param['theta2']
        
        # Function for theta1 value computation
        # CHECK TO IMPROVE THIS FUNCTION

        lt = lr0 + rp*q2 + rp*theta2
        theta_s = np.arctan(lt/rp) - theta2
        rs = np.sqrt(lt**2 + rp**2)
        zs = d1 - rs*np.sin(theta_s)

        sol_theta1 = np.arccos((-zs-l1*np.cos(q1))/l2) + q1

        # Re-assignment of values
        self.leg_param['theta1_p'] = self.leg_param['theta1']
        self.leg_param['theta1'] = sol_theta1
    
    # ******************* INVERSE KINEMATIC ***********************    
    def inverse_kinematics(self, vel_B, time_step = 0.02):
        # This methods run through the inverse kinematic chain
        # INPUT: 
        #       vel_B: (2,) numpy vector|| velocity from motion planner 
        #       time_step: int          || integration timestep
        # OUTPUT: 
        #       q_val: (2,) numpy vector || new servo positions 

        # Run through kinematic compute chain for front leg
        self.compute_ik_angle1(vel_B)
        self.compute_q2_dot()
        self.explicit_ik_update(time_step)

        # Reassign previous timestep values
        self.leg_param['yH_dot_p'] = self.leg_param['yH_dot']
        self.leg_param['zH_dot_p'] = self.leg_param['zH_dot']
        # Assign new values
        self.leg_param['yH_dot'] = vel_B[0]
        self.leg_param['zH_dot'] = vel_B[1]

        return np.array([self.leg_param['q1'], self.leg_param['q2']])

    def compute_ik_angle1(self, vel_B):
        # This methods calculates the q1_dot and theta1_dot as a pre-step
        # INPUT: 
        #       vel_B: (2,) numpy vector|| velocity from motion planner 
        # OUTPUT: none

        # Unload necessary equation parameters
        l1 = self.leg_param['l1']
        l3 = self.leg_param['l3']
        l4 = self.leg_param['l4']

        q1 = self.leg_param['q1']
        theta1 = self.leg_param['theta1']
        theta3 = self.leg_param['theta3']    
        
        # Jacobian from forward kinematics
        ik_jac = np.array([[l1*np.cos(q1)+l3*np.cos(theta1 - q1)+l4*np.cos(theta3+theta1-q1), -l3*np.cos(theta1 - q1)-l4*np.cos(theta3+theta1-q1)],
                           [l1*np.sin(q1)-l3*np.sin(theta1 - q1)-l4*np.sin(theta3+theta1-q1), l3*np.sin(theta1 - q1)+l4*np.sin(theta3+theta1-q1)]])


        sol_q_dot = np.linalg.solve(ik_jac,vel_B)

        # *************************************************
        # OPTION: implement a maximum velocity control here
        q1_dot = sol_q_dot[0]
        theta1_dot = sol_q_dot[1]

        # Reassign previous timestep values
        self.leg_param['q1_dot_p'] = self.leg_param['q1_dot']
        self.leg_param['theta1_dot_p'] = self.leg_param['theta1_dot']
        # Assign new values
        self.leg_param['q1_dot'] = q1_dot
        self.leg_param['theta1_dot'] = theta1_dot

    def compute_q2_dot(self):
        # This methods computes q2_dot using the values from the previous step
        # INPUT: none
        # OUTPUT: none

        # Unload necessary equation parameters
        l1 = self.leg_param['l1']
        l2 = self.leg_param['l2']
        rp = self.leg_param['rp']
        d1 = self.leg_param['d1']
        lr0 = self.leg_param['lr0']
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']
        theta2 = self.leg_param['theta2']
        q1_dot = self.leg_param['q1_dot']
        theta1_dot = self.leg_param['theta1_dot']

        # Symbolic solution from Matlab
        sol_q2_dot = -(theta1_dot + q1_dot*((l1*np.sin(q1))/(l2*np.sqrt(1 - (d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1))**2/l2**2)) - 1) + (q1_dot*(2*l1*np.sin(q1)*(d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1)) - 2*l1*np.cos(q1)*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.sin(q1)))*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2)))/(l2*np.sqrt(1 - (d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1))**2/l2**2)*(2*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.sin(q1))*(np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + (rp*np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2)) - 2*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))*(d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1)))))/(((np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))/(l2*np.sqrt(1 - (d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1))**2/l2**2)) + ((2*((np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))*(d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1)) - 2*((np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) + (rp*np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.sin(q1)))*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2)))/(l2*np.sqrt(1 - (d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1))**2/l2**2)*(2*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.sin(q1))*(np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + (rp*np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2)) - 2*(np.cos(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(1/((lr0 + q2*rp + rp*theta2)**2/rp**2 + 1) - 1)*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) - (rp*np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*(lr0 + q2*rp + rp*theta2))/np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2))*(d1 + np.sin(theta2 - np.arctan((lr0 + q2*rp + rp*theta2)/rp))*np.sqrt((lr0 + q2*rp + rp*theta2)**2 + rp**2) + l1*np.cos(q1)))))
        # *************************************************
        # OPTION: implement a maximum velocity control here

        # Re-assignment of values
        self.leg_param['q2_dot_p'] = self.leg_param['q2_dot']
        self.leg_param['q2_dot'] = sol_q2_dot

    def explicit_ik_update(self, time_step):
        # This methods performs explicit Euler update to compute q1 and q2
        # INPUT: none
        # OUTPUT: none

        # Loading of relevant values
        q1 = self.leg_param['q1']
        q2 = self.leg_param['q2']
        q1_dot = self.leg_param['q1_dot']
        q2_dot = self.leg_param['q2_dot']
        
        # Implicit Euler Update Step
        # *************************************************
        # OPTION: implement a maximum velocity control here
        q1_new = q1 + time_step * q1_dot
        q2_new = q2 + time_step * q2_dot

        # print("Servo velocity q1: {}".format(q1_dot))
        # print("Servo velocity q2: {}".format(q2_dot))

        # Re-assignment of values
        self.leg_param['q1_p'] = self.leg_param['q1']
        self.leg_param['q2_p'] = self.leg_param['q2']

        self.leg_param['q1'] = q1_new
        self.leg_param['q2'] = q2_new