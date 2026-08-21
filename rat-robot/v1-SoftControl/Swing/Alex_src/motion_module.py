#!/usr/bin/env python3
# *******************************************************
# Type: Motion controller
# 
# Motion controller for the mouse
# Handles state machine and low level spine and leg control.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 26.04.2021
# *********************************************************
import argparse
# Import the leg and motionplanner modules
from mouse_controller.leg_controller import Leg_Controller
from mouse_controller.state_machine.leg_state_machine import Leg_State_Machine
from mouse_controller.mouse_parameters_dir import Gait_Parameters, Mouse_Parameters
from time import sleep

# Import other relevant libraries for ROS
import numpy as np

from time import sleep

from numpy.core.numerictypes import maximum_sctype

from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt

class Motion_Module:
    def __init__(self,fre):
        self.model_name = "dynamic_4l_t3.xml"
        #self.model_name = "dynamic_4l_t3_no_stl.xml"

        self.model_path = "../models/"+self.model_name
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        self.legPosName = [
            ["router_shoulder_fl", "foot_s_fl"],
            ["router_shoulder_fr", "foot_s_fr"],
            ["router_hip_rl", "foot_s_rl"],
            ["router_hip_rr", "foot_s_rr"]]
        self.legRealPoint_x = [[],[],[],[]]
        self.legRealPoint_y = [[],[],[],[]]
        self.trgXList = [[],[],[],[]]
        self.trgYList = [[],[],[],[]]

        self.init_mouse_variables(fre)
        self.init_controllers()
        self.main()
        
    def init_mouse_variables(self, fre):
        self.gait_parameters2 = Gait_Parameters()
        self.mouse_parameters = Mouse_Parameters()
        self.general_st_parameters2 = self.gait_parameters2.st_trot_parameters
        self.general_st_parameters2["cycle_freq"] = fre 
        self.front_leg_parameters2 = self.gait_parameters2.st_trot_param_f
        self.rear_leg_parameters2 = self.gait_parameters2.st_trot_param_r

    def init_controllers(self):
        # Initialize the key components of the motion module
        # Spine modes:
        # 0: purely turning motion, nothing else
        # 1: turning motion + spine modulation
        # 2: turning motion + balance mode (balance mode for 2 and 3 leg contact)
        self.spine_mode = 0
        self.offset_mode = False
        self.balance_mode = True
        self.fsm = Leg_State_Machine(self.general_st_parameters2)
        self.leg_controller = Leg_Controller(self.gait_parameters2, self.mouse_parameters)
        self.vel_in = 0.0
        self.turn_rate = 0.0
        self.buttons = [0]*4
        self.prev_buttons = [0]*4
        self.leg_n = 0

    def motion_node(self,rate):
        # main starter method
        dry = 0

        self.fsm.timer.reset_times()
        sleep(0.002)
        self.offset_mode = True
        self.leg_n = (self.leg_n+1)%4
        for i in range(6000):
            self.vel_in = 0.4
            self.turn_rate = 0
            vel = self.vel_in * np.ones((4,))
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()
            # Steps of the full controller to generate values
            target_leg_positions, q_legs, q_spine = \
                self.leg_controller.run_controller(leg_states, leg_timings, 
                    norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            q_tail = 0#self.tail_extension(norm_time, self.vel_in)

            self.gen_messages(target_leg_positions, q_legs, q_spine, q_tail)


        self.offset_mode
    def tail_extension(self, timing: float,vel: float, offset=0, scaling=0.5) -> float:
        # THis function helps extend the spine stride during gait.
        # Timing value: is normalized time value [0,1]
        scale = min(4.5*np.abs(vel)**2, scaling)
        q_tail = scale*np.cos(2*np.pi*timing+offset)
        return q_tail

    def gen_messages(self, target_leg_positions, q_legs, q_spine, q_tail):
        target_leg_positions.astype(dtype=np.float32)
        #print(target_leg_positions)
        q_values = np.concatenate((q_legs,np.array(([q_tail,0,0,q_spine]))))
        q_values.astype(dtype=np.float32)

        self.sim.data.ctrl[:] = q_values
        self.sim.step()
        self.viewer.render() 

        for i in range(4):
            originPoint = self.sim.data.get_site_xpos(self.legPosName[i][0])
            currentPoint = self.sim.data.get_site_xpos(self.legPosName[i][1])
            #print(originPoint, currentPoint)
            tX = currentPoint[1]-originPoint[1]
            tY = currentPoint[2]-originPoint[2]
            self.legRealPoint_x[i].append(tX)
            self.legRealPoint_y[i].append(tY)

            self.trgXList[i].append(target_leg_positions[i][0])
            self.trgYList[i].append(target_leg_positions[i][1])
        
    def main(self):
        self.motion_node(rate=80)

        fig, axs = plt.subplots(2,2)
        subTitle = ["Fore Left Leg", "Fore Right Leg",
            "Hind Left Leg", "Hind Right Leg"]
        for i in range(4):
            pos_1 = int(i/2)
            pos_2 = int(i%2)
            #print(pos_1, pos_2)
            axs[pos_1,pos_2].set_title(subTitle[i])
            axs[pos_1,pos_2].plot(self.trgXList[i][500:], self.trgYList[i][500:]
                , label='Target trajectory')
            axs[pos_1,pos_2].plot(self.legRealPoint_x[i][500:], self.legRealPoint_y[i][500:]
                , label='Simulation trajectory')
        fig.suptitle('Control trajectory of model from Alex', fontsize=15)
        plt.subplots_adjust(hspace=0.5,wspace=0.3)
        plt.savefig('alex_traj.png')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Description.")
    parser.add_argument('--fre', default=0.67,
        type=float, help="Gait stride")
    args = parser.parse_args()
    Motion_Module(args.fre)
