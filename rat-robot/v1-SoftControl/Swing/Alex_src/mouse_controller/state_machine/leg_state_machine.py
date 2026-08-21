# *******************************************************
# Type: Leg State Machine
# 
# State determinor takes in the gait parameters and 
# runs a time based state machine to switch each
# leg between stance & swing phase.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

import numpy as np
from mouse_controller.state_machine.time_machine import Time_Machine
from mouse_controller.state_machine.state_determiner import State_Determinor

class Leg_State_Machine:

    def __init__(self, gait_parameters):
        print("Initializing the leg state machine")
        # Leg states vector     1: stance       0: swing
        self.leg_states = np.zeros((4,),dtype=int)
        self.timer = Time_Machine(gait_parameters['cycle_freq'])
        self.state_determinor = State_Determinor(gait_parameters)

        # Checks for the first time run that the timings are correct
        self.leg_phase_timings = self.state_determinor.neutral_times_phases*self.state_determinor.stance_timings

    def run_state_machine(self):    
        self.timer.update_time()
        self.compute_normalized_time_units()
        self.composite_H(self.norm_time_vector)
        self.compute_normalized_phase_timings()
        return (self.leg_states, self.leg_phase_timings_norm, self.normalized_time)

    def compute_normalized_time_units(self):
        normalized_delta_time = self.timer.delta_time / self.state_determinor.cycle_time
        self.normalized_time = self.timer.curr_cycle_time / self.state_determinor.cycle_time
        #print("Normalized time: {} || Normalized delta time: {}".format(self.normalized_time,normalized_delta_time))
        self.norm_time_vector = self.normalized_time*np.ones((4,))
        self.norm_delta_time = normalized_delta_time
    
    def compute_normalized_phase_timings(self):
        # Computing the current normalized time for the legs in their current gait phase

        leg_states_2d = np.array([1-self.leg_states,self.leg_states]).T
        self.leg_phase_timings = leg_states_2d*((self.leg_phase_timings + self.norm_delta_time))
        self.leg_phase_timings_norm = self.leg_phase_timings/self.state_determinor.stance_timings
        
    def update_time_machine(self, cycle_freq):
        self.timer.update_cycle_time(cycle_freq)

    def update_determinor_machine(self, gait_parameters):
        self.state_determinor.update_gait_parameters(gait_parameters)

    def composite_H(self,t):
        # Combined heavyside functions for gaits
        c1 = self.state_determinor.gait_regions[:,0]
        c2 = self.state_determinor.gait_regions[:,1]
        c3 = self.state_determinor.gait_regions[:,2]
        self.leg_states = (np.round(self.H_f(c1,t)-self.H_f(c2,t)+self.H_f(c3,t))).astype(int)
        #print(self.leg_states)

    def get_current_cycle_time(self):
        return self.timer.curr_cycle_time

    def get_delta_cycle_time(self):
        return self.timer.delta_time

    def H_f(self,c,x,a=1,b=1):
        # Step function (heavyside) function, providing output
        return 0.5*(np.sign(x-c)+b)*a
