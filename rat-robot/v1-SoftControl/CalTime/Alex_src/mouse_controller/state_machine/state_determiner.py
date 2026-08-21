import numpy as np

# *******************************************************
# Type: State Machine Subscript
# 
# State determinor takes in the gait parameters and 
# runs a time based state machine to switch each
# leg between stance & swing phase.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

class State_Determinor:
    # Keep gait regions as percentage of the gait cycle

    gait_regions = np.array([[-0.1, 0.0, 1.1],
                             [-0.1, 0.0, 1.1],
                             [-0.1, 0.0, 1.1],
                             [-0.1, 0.0, 1.1]])

    def __init__(self, gait_parameters):
        self.update_gait_parameters(gait_parameters)

    def update_cycle_time(self, cycle_freq):
        # If a new leg frequency is supplied, update all timing based values
        # Currently not relevant [as state machine uses normalized time]

        self.cycle_freq = cycle_freq
        self.cycle_time = 1.0/cycle_freq
        self.stance_time = self.cycle_distr*self.cycle_time
    
    def update_gait_parameters(self, gait_parameters):
        self.cycle_distr = gait_parameters['cycle_distr']
        self.leg_cycle_offset = gait_parameters['leg_cycle_offset']
        self.cycle_freq = gait_parameters['cycle_freq']
        self.cycle_time = 1.0/self.cycle_freq
        self.stance_time = self.cycle_distr*self.cycle_time
        self.compute_gait_regions()

    def compute_gait_regions(self):
        # Start and end gait % for the stance phase
        # Converted into a range matrix (4,3) to be used
        # for the step function in the state machine.

        # Padded gait region array
        self.gait_regions = np.array([[-0.1, 0.0, 1.1],
                                        [-0.1, 0.0, 1.1],
                                        [-0.1, 0.0, 1.1],
                                        [-0.1, 0.0, 1.1]])
        end = (self.leg_cycle_offset + self.cycle_distr)%1.0
        ranger = np.stack((self.leg_cycle_offset,end), axis=-1)

        # Timing regions for the individual legs
        self.leg_timing_regions = ranger
        # Total time for each phase of the cycle (needed for time normalization)
        self.stance_timings = np.array([(1-self.cycle_distr),self.cycle_distr])
        comparison = (ranger[:,0] > ranger[:,1]).astype(int)
        ranger.sort(axis=1)

        for i in range(4):
            self.gait_regions[i,comparison[i]:(comparison[i]+2)] = ranger[i]
        
        gait_exp = np.stack((end,self.leg_cycle_offset),axis=-1)+self.stance_timings
        gait_exp2 = gait_exp%1.0
        gait_exp3 = -1*gait_exp2/self.stance_timings
        self.neutral_times_phases = gait_exp3 + 1
        # print(self.gait_regions)
