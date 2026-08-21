import numpy as np
import rospkg
from mouse_controller.spine_models.spine_kinematics import Spine_Kinematics, Spine_Helpers

from mouse_controller.trajectory_generator.trajectory_generator import Leg_Trajectory_Generator

from mouse_controller.leg_unit_class import Leg_Unit
from mouse_controller.mouse_parameters_dir import Gait_Parameters, Mouse_Parameters
from mouse_controller.spine_controller import Spine_Controller

class Leg_Controller:

    def __init__(self, gait_parameters, mouse_parameters):
        self.gait_parameters = gait_parameters
        self.geometric_param = mouse_parameters.mouse_geometric_param
        self.ik_legs = Inverse_Leg_Kinematics()
        self.spine_controller = Spine_Controller(self.geometric_param)
        self.setup_trajectory_generators()
        self.previous_leg_states = -1*np.ones((4,),dtype = int)
        print("Initialized leg controller")

    def setup_trajectory_generators(self):
        front_leg_parameters = self.gait_parameters.st_trot_param_f
        rear_leg_parameters = self.gait_parameters.st_trot_param_r

        fl_traj = Leg_Trajectory_Generator(front_leg_parameters,0)
        fr_traj = Leg_Trajectory_Generator(front_leg_parameters,1)
        rl_traj = Leg_Trajectory_Generator(rear_leg_parameters,2)
        rr_traj = Leg_Trajectory_Generator(rear_leg_parameters,3)

        self.traj_obj = {0: fl_traj,
                         1: fr_traj,
                         2: rl_traj,
                         3: rr_traj}

    def update_leg_traj_param(self, gait_param):
        #Empty for now
        return

    def run_controller(self, leg_states, leg_timings, norm_time, leg_velocities, turn_rate, spine_mode, offset_mode):
        alphas = self.compute_turn_alphas_exponentially(turn_rate)
        #print("Alpha values: {}".format(alphas))

        next_leg_positions = self.compute_next_leg_positions(leg_states, leg_timings, leg_velocities, alphas)

        # Only for testing purposes
        current_leg_positions = self.ik_legs.get_current_leg_positions()
        if True:
            q_spine, leg_offsets = self.spine_controller.run_controller(turn_rate, current_leg_positions,leg_states, norm_time,leg_velocities[0], spine_mode, offset_mode)

            next_leg_positions[2,1] = next_leg_positions[2,1] + leg_offsets[0]
            next_leg_positions[3,1] = next_leg_positions[3,1] + leg_offsets[1]
        else:
            q_spine = 0.0
        
        leg_q_values = self.ik_legs.run_inverse_leg_kinematics(next_leg_positions)

        return (next_leg_positions, leg_q_values, q_spine) 

    def compute_turn_alphas_linear(self, turn_rate):
        # In here we compute alpha value adjustments for specified turn rates
        # TO-DO
        alpha_left = min(1-turn_rate,1)
        alpha_right = min(1+turn_rate,1)
        alphas = np.array([alpha_left, alpha_right, alpha_left, alpha_right])
        alphas = np.ones((4,))
        return alphas

    def compute_turn_alphas_log(self,turn_rate):
        alpha_left = min(1-np.sign(turn_rate)*(1/np.log(2))*np.log(np.abs(turn_rate)+1),1)
        alpha_right = min(1+np.sign(turn_rate)*(1/np.log(2))*np.log(np.abs(turn_rate)+1),1)
        alphas = np.array([alpha_left, alpha_right, alpha_left, alpha_right])
        alphas = np.ones((4,))
        return alphas

    def compute_turn_alphas_exponentially(self, turn_rate):
        """Compute the turn alpha offsets for each leg"""
        alpha_left = min(1-turn_rate**5,1)
        alpha_right = min(1+turn_rate**5,1)
        alphas = np.array([alpha_left, alpha_right, alpha_left, alpha_right])
        # alphas = np.ones((4,))
        return alphas

    def compute_new_trajectory(self, leg_velocities, turn_rates):
        return

    def compute_next_leg_positions(self, leg_states, leg_timings, leg_velocities, alphas):
        status_change = np.abs(leg_states-self.previous_leg_states)
        # print("Status change vector: ")
        next_leg_positions = np.zeros((4,2))
        
        # Remember, the legs are setup as follows
        # 0: front left
        # 1: front right
        # 2: rear left
        # 3: rear right
        for i in range(4):
            if status_change[i] != 0:
                self.traj_obj[i].new_trajectory_compute(leg_velocities[i],leg_states[i],alphas[i])
            next_leg_positions[i,:] = self.traj_obj[i].next_leg_point(leg_timings[i,leg_states[i]])
        
        self.previous_leg_states = leg_states

        return next_leg_positions

class Stance_Phase_State:

    def __init__(self):

        return

class Swing_Phase_State:
    
    def __init__(self):

        return

class Inverse_Leg_Kinematics:
    # Class that handles the IK aspects of the legs

    def __init__(self):
        print("Initialized inverse leg kinematics")
        self.mouse_parameters = Mouse_Parameters()
        self.setup_leg_models()

    def setup_leg_models(self):
        front_leg_t3_param = self.mouse_parameters.fr_t1_param
        rear_leg_t3_param = self.mouse_parameters.rr_t3_param

        self.lu_fl = Leg_Unit('fr3',front_leg_t3_param)
        self.lu_fr = Leg_Unit('fr3',front_leg_t3_param)
        self.lu_rl = Leg_Unit('rr3',rear_leg_t3_param)
        self.lu_rr = Leg_Unit('rr3',rear_leg_t3_param)

    def run_inverse_leg_kinematics(self,new_target_leg_positions,timer=0.11):
        # Todo 
        # Replace the timer value with the actual timer
        # Check whether such an implementation actual works
        # print("Compute the necessary q_values for our leg")
        difference = self.compute_pos_difference(new_target_leg_positions)
        # Velocity passed to the inverse kinematics of the legs must be (2,)
        q_values_legs = self.compute_inverse_kinematics(difference,timer)

        return q_values_legs

    def compute_pos_difference(self, new_target_leg_positions):
        # Note that the current_leg_position() returns a (3,) vector of (x,y,z)
        # We only care about (y,z) so need to slice the vector/matrix
        current_leg_positions = self.get_current_leg_positions()
        #print("Current leg positions: {}".format(current_leg_positions))
        #print("Target leg positions: {}",format(new_target_leg_positions))
        difference = new_target_leg_positions - current_leg_positions[:,1:]

        return difference

    def compute_inverse_kinematics(self, difference,timer=0.001):
        vel = difference
        #print("Positional difference of target and goal: {}".format(vel))
        current_q_values = self.internal_q_value_return()
        #print("Parmeters for the first leg:")
        #print("Current q_values of the leg: {}".format(current_q_values[0,:]))
        q_values_fl = self.lu_fl.kinematic_update_no_mp(vel[0,:],current_q_values[0,:],timer)
        q_values_fr = self.lu_fr.kinematic_update_no_mp(vel[1,:],current_q_values[1,:],timer)
        # q_values_fl = np.array([0,0])
        # q_values_fr = np.array([0,0])
        q_values_rl = self.lu_rl.kinematic_update_no_mp(vel[2,:],current_q_values[2,:],timer)
        q_values_rr = self.lu_rr.kinematic_update_no_mp(vel[3,:],current_q_values[3,:],timer)
        # print("Elapsed total sim time: {}".format(time.perf_counter() - time_start))

        return np.concatenate((q_values_fl, q_values_fr,q_values_rl,q_values_rr))


    def internal_q_value_return(self):
        # Placeholder function until feedback from the q-values arrives
        current_q_values = np.array([self.lu_fl.current_leg_servos(),
                                    self.lu_fr.current_leg_servos(),
                                    self.lu_rl.current_leg_servos(),
                                    self.lu_rr.current_leg_servos()])
        
        return current_q_values

    def get_current_leg_positions(self):
        current_leg_pos = np.array([self.lu_fl.current_leg_position(),
                                    self.lu_fr.current_leg_position(),
                                    self.lu_rl.current_leg_position(),
                                    self.lu_rr.current_leg_position()])
        return current_leg_pos