import numpy as np
from time import perf_counter

# *******************************************************
# Type: State Machine Subscript
# 
# Time tracker module for keeping gait timings.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 05.06.2021
# *********************************************************

class Time_Machine:
    # 
    
    def __init__(self, cycle_freq):
        # Set the time the start of the time
        self.curr_cycle_time = 0
        self.start_time = perf_counter()
        self.cycle_start = perf_counter()
        self.cycle_time = 1.0/cycle_freq
        self.delta_time = 0.0
        self.last_time = 0.0

    def update_cycle_time(self, cycle_freq):
        # This is the gait cycle period (1/cycle_time)
        self.cycle_time = 1.0/cycle_freq

    def update_time(self):
        now_time = perf_counter()
        current_time = now_time - self.cycle_start
        self.delta_time = current_time - self.last_time
        self.last_time = current_time
        self.curr_cycle_time = round(current_time % self.cycle_time,3)
        # For testing purposes
        # print(self.curr_cycle_time)
    
    def reset_times(self):
        self.cycle_start = perf_counter()
        self.delta_time = 0.0
        self.last_time = 0.0
