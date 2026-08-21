import numpy as np

# *******************************************************
# Type: Data Directory
# 
# Contains gait & mouse specific standard parameters
#
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************


class Gait_Parameters:

    def __init__(self):
        self.set_trot_parameters()


    def set_trot_parameters(self):
        # High level parameters for the trot gait
        # Cycle_freq       [Hz]        float           - time for a full cycle
        # cycle_distr       [/]         float           - stance % vs. swing % of cycle
        # leg_cycle_offset  [/]         float⁽⁴⁾        - gait offset (start of stance within cycle)
        # Legs: 0, 1, 2, 3| FL, FR, RL, RR
        self.st_trot_parameters = {'cycle_freq': 0.67,
                                'cycle_distr': 0.5,
                                'leg_cycle_offset': np.array([0.5,0,0,0.5])}
        
        self.st_trot_parameters1 = {'cycle_freq': 0.67,
                                'cycle_distr': 0.68,
                                'leg_cycle_offset': np.array([0.63,0.14,0,0.54])}
        
        self.gallop_rot_parameters = {'cycle_freq': 1.0,
                                'cycle_distr': 0.3,
                                'leg_cycle_offset': np.array([0.65,0.55,0,0.1])}

        self.walk_trot_parameters = {'cycle_freq': 0.5,
                                'cycle_distr': 0.6,
                                'leg_cycle_offset': np.array([0.5,0,0,0.5])}
        
        # Due to unsymmetric front/rear legs different in neutral heights of stance
        self.st_trot_param_f = {'cycle_freq':           self.st_trot_parameters['cycle_freq'],
                                'amp_swing':            0.01,
                                'amp_stance':           0.005,
                                'max_stride_length':    0.02,
                                'neutral_stance_pos':   -0.045,
                                'neutral_stride_pos':   -0.005,
                                'cycle_distr':          self.st_trot_parameters['cycle_distr'],
                                'leg_cycle_offset':     self.st_trot_parameters['leg_cycle_offset']}

        self.st_trot_param_r = {'cycle_freq':           self.st_trot_parameters['cycle_freq'],
                                'amp_swing':            0.01,
                                'amp_stance':           0.005,
                                'max_stride_length':    0.02,
                                'neutral_stance_pos':   -0.05,
                                'neutral_stride_pos':   0.0,
                                'cycle_distr':          self.st_trot_parameters['cycle_distr'],
                                'leg_cycle_offset':     self.st_trot_parameters['leg_cycle_offset']}

class Mouse_Parameters:

    def __init__(self):
        self.set_mouse_parameters()

    def set_mouse_parameters(self):
        # Setting leg parameters - units [m]
        self.fr_t1_param = {'lr0':0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295, 
                            'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145,'theta3':23*np.pi/180}
        
        self.rr_t3_param = {'lr0':0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317, 
                            'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205,'theta3':73*np.pi/180}

        # Setting general mouse parameters
        # [x,y,z] - units [m]
        # ^z -> y (RHS)
        self.mouse_geometric_param = {'fl_shoulder':  np.array([ 0.0368, -0.0863,  0.0024]),
                    'fr_shoulder':  np.array([-0.0368, -0.0863,  0.0024]),
                    'rl_hip':       np.array([ 0.0428,  0.0356,  0.0095]),
                    'rr_hip':       np.array([-0.0428,  0.0356,  0.0095]),
                    'com_front':    np.array([ 0.0000, -0.0565,  0.0078]),
                    'com_rear':     np.array([ 0.0000,  0.0578,  0.0034]),
                    'spine_start':  np.array([ 0.0000, -0.0248,  0.0348]),
                    'spine_end':    np.array([ 0.0000,  0.0268,  0.0194]), #Crosscheck with kinematic chain
                    'mass_front':   0.1537,
                    'mass_rear':    0.0435,
                    'legs':         0.0020,
                    'head':         0.0231,
                    'spine':        0.0040}

        # Initialize new coordinates relative to the FRONT COM and the REAR COM (for the rear components)
        # This means that the FRONT COM becomes -> [0.0, 0.0, 0.0]
        self.mouse_geometric_param['fl_com_shoulder'] = self.mouse_geometric_param['fl_shoulder'] - self.mouse_geometric_param['com_front'] 
        self.mouse_geometric_param['fr_com_shoulder'] = self.mouse_geometric_param['fr_shoulder'] - self.mouse_geometric_param['com_front'] 
        self.mouse_geometric_param['rl_com_hip'] = self.mouse_geometric_param['rl_hip'] - self.mouse_geometric_param['com_front'] 
        self.mouse_geometric_param['rr_com_hip'] = self.mouse_geometric_param['rr_hip'] - self.mouse_geometric_param['com_front'] 
        self.mouse_geometric_param['com_spine'] = self.mouse_geometric_param['spine_start'] - self.mouse_geometric_param['com_front']
        self.mouse_geometric_param['com_comr'] = self.mouse_geometric_param['com_rear'] - self.mouse_geometric_param['com_front']
        self.mouse_geometric_param['com_spine_end_neutral'] = np.array([0.0, 0.0834, 0.0116])
        self.mouse_geometric_param['se_comr'] = self.mouse_geometric_param['com_comr'] - self.mouse_geometric_param['com_spine_end_neutral']
        self.mouse_geometric_param['se_rl'] = self.mouse_geometric_param['rl_com_hip'] - self.mouse_geometric_param['com_spine_end_neutral']
        self.mouse_geometric_param['se_rr'] = self.mouse_geometric_param['rr_com_hip'] - self.mouse_geometric_param['com_spine_end_neutral']
        self.mouse_geometric_param['se'] = np.array([0.0, 0.0, 0.0])
