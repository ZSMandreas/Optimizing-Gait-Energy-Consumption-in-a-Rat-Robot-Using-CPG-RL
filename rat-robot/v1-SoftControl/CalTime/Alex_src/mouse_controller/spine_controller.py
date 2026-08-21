import numpy as np
from numpy.core.arrayprint import _leading_trailing

class Spine_Controller:

    def __init__(self, geometric_param):
        print("Intializing spine controller")
        self.spine_angle = 0.0
        self.servo_angle = 0.0
        self.spine_distance = 0.0
        self.geometric_param = geometric_param
        # Geometric lenghs of the spine 
        self.spine_helper_functions = Spine_Kinematic_Helpers()
        # Maximum spine flexion
        self.d_spine_max = 0.009
        self.T_total_0 = self.compute_spine_transform(0.0)

    def run_controller(self, turn_rate: float, leg_positions: np.ndarray, leg_states: np.ndarray, timing: float, vel: float, mode = 0, offset_mode=False):
        # Several spine modes available
        # 0: purely turning motion, nothing else
        # 1: turning motion + spine modulation
        # 2: turning motion + balance mode (balance mode for 2 and 3 leg contact)

        # 1. Compute the necessary servo angle (from our special modes)
        servo_offset = self.compute_servo_offset_angle(leg_positions, leg_states, timing, vel, mode)
        # 2. Pass forward to compute the overall angle
        self.servo_angle = (turn_rate)*0.4 + (1-np.abs(turn_rate))*servo_offset
        self.spine_angle = self.spine_helper_functions.servo_to_spine_angle(self.servo_angle)
        # 3. Compute the necessary leg offsets
        self.leg_offsets = self.compute_leg_offset(self.servo_angle, leg_positions, offset_mode)

        return (self.servo_angle, self.leg_offsets)
    
    def compute_servo_offset_angle(self, leg_positions: np.ndarray, leg_states: np.ndarray, timing: float, vel: float, mode: int) -> float:
        # First compute the necessary servo angle
        if mode == 1:
            # Compute the servo angle from the mode
            servo_offset = self.spine_stride_extension(timing, vel)
        elif mode == 2:
            # Compute servo angle in leg balance mode
            servo_offset = self.balance_control(leg_positions, leg_states)
        else:
            servo_offset = 0.0
        
        return servo_offset

    def compute_leg_offset(self, servo_angle: float, leg_positions: np.ndarray, offset_mode: bool):
        # Computes the offset values z-values induced by spine flexion
        # 
        # Output: tuple (2) of type float (offset for rl, rr)
        if offset_mode:
            T_total = self.compute_spine_transform(servo_angle)
            leg_offsets = self.compute_leg_offset_values(T_total, leg_positions)
        else:
            leg_offsets = (0.0,0.0)
        
        return leg_offsets

    def compute_spine_transform(self, servo_angle: float) -> np.ndarray:
        # Computes the transformation matrix of the kinematic chain
        # Induced by the chosen servo angle.
        # self.spine_angle = self.spine_helper_functions.servo_to_spine_angle(servo_angle)
        DH_params = self.spine_helper_functions.compute_dh_table(self.spine_angle)
        T_total = self.spine_helper_functions.compute_transform(DH_params)
        return T_total

    def compute_leg_offset_values(self, T_total: np.ndarray, leg_positions: np.ndarray):
        # Leg positions: FL, FR, RL, RR
        # 
        leg_pos_rl = leg_positions[2,:] + self.geometric_param['se_rl']
        leg_pos_rr = leg_positions[3,:] + self.geometric_param['se_rr']

        rear_block = np.ones((4,6))

        # Contains the most important elements of the rear leg
        # Rear COM, rear hip (left & right), rear footend (left and right), rear spine_end
        # Matrix size: 4x6
        rear_block[:-1,:] = (np.array([self.geometric_param['se_comr'],
                                self.geometric_param['se_rl'],
                                self.geometric_param['se_rr'],
                                leg_pos_rl,
                                leg_pos_rr,
                                self.geometric_param['se']])).T

        com_spine_pos = np.zeros((4,))
        com_spine_pos[:-1] = self.geometric_param['com_spine']
        # This operation shifts all elements to be relative within the whole mouse model
        # Matrix size 6x4
        rear_block_t = (np.dot(T_total,rear_block)).T
        rear_block_z = (np.dot(self.T_total_0,rear_block)).T
        offset_rl = rear_block_z[3,2] - rear_block_t[3,2]
        offset_rr = rear_block_z[4,2] - rear_block_t[4,2]

        # rear_block_zeroed = (rear_block + com_spine_pos)[:,:-1]

        return (offset_rl,offset_rr)

    def spine_stride_extension(self, timing: float,vel: float, offset=0, scaling=0.4) -> float:
        # THis function helps extend the spine stride during gait.
        # Timing value: is normalized time value [0,1]
        scale = min(4.5*np.abs(vel)**2, scaling)
        q_spine = scale*np.cos(2*np.pi*timing+offset)
        return q_spine

    def balance_control(self, leg_positions: np.ndarray, leg_states: np.ndarray) -> float:
        # Method for running the balance based controller
        # First we determine which legs are in contact 
        # From this, we can determine the diagonal points. 
        # We overall have six cases that need to be determined. 
        d_pair, d_sign = self.state_determinor(leg_states)
        # Once the case is determined, we can generate the target goal distance
        if type(d_pair) == tuple:
            d_target = d_sign*self.d_spine_max
            front_leg = leg_positions[d_pair[0]]
            front_shoulder = self.geometric_param[self.spine_helper_functions.aux_map[d_pair[0]]]
            rear_leg = leg_positions[d_pair[1]]
            rear_hip = self.geometric_param[self.spine_helper_functions.aux_map[d_pair[1]]]
            coords = np.concatenate((front_leg, front_shoulder, rear_leg, rear_hip))
            q_spine_servo = self.compute_balance_spine_angle(coords, d_target)
        else:
            q_spine_servo = 0.0
        # Then we run the forward pass to obtain the current distance
        # Followed by the backward pass to obtain the required spine value
        return q_spine_servo

    def compute_balance_spine_angle(self, coords: np.ndarray, d_target) -> float:
        # In here compute the servo angle given the diagonal and target distance of the COM to spine
        current_spine_angle = self.spine_angle
        d_current = self.spine_helper_functions.spine_stability_forward_pass(current_spine_angle,coords)
        d_delta = d_target - d_current
        q_spine_target = self.spine_helper_functions.spine_stability_backward_pass(current_spine_angle,coords, d_delta)
        q_spine_servo = self.spine_helper_functions.spine_to_servo_angle(q_spine_target)
        return q_spine_servo

    def state_determinor(self, leg_states: np.ndarray): 
        # Return the diagonal pair and sign of the distance
        # If not applicable, return (-1, -1) tuple
        state_set = str(leg_states[0]) + str(leg_states[1])+str(leg_states[2]) + str(leg_states[3])
        d_pair = self.spine_helper_functions.d_pair_map[state_set]
        if d_pair[0] == -1:
            return (-1, -1 )
        else:
            d_sign = self.spine_helper_functions.d_map[state_set]
        
        return (d_pair, d_sign)

class Spine_Kinematic_Helpers:
    # This class contains a set of useful helper functions to compute transform matrices, DH tables etc.
    # Used by the Spine_Controller.

    def __init__(self):
        print("Spine helper class initiated")
        self.spine_lengths = np.array([0.0016,0.0070,0.0092,0.0014])
        self.d_map =    {'-1': 0,
                        '1001': 0,
                        '1101': -1,
                        '1011': 1,
                        '0110': 0,
                        '0111': -1,
                        '1110': 1,
                        '1111': 0}
        self.d_pair_map = {'-1': (-1,-1),
                            '0000': (-1,-1),
                            '1000': (-1,-1),
                            '0100': (-1,-1),
                            '0010': (-1,-1),
                            '0001': (-1,-1),
                            '1001': (0,3),
                            '1101': (0,3),
                            '1011': (0,3),
                            '0110': (1,2),
                            '0111': (1,2),
                            '1110': (1,2),
                            '1111': (-1,-1),
                            '1100': (-1,-1),
                            '0011': (-1,-1),
                            '1010': (-1,-1),
                            '0101': (-1,-1)}
        self.aux_map = {0: 'fl_com_shoulder',
                        1: 'fr_com_shoulder',
                        2: 'se_rl',
                        3: 'se_rr'}
                        
        # Placing a maximum on the spine angle for stability
        self.theta_s_max = 0.1
        self.theta_s_min = -0.1

    def compute_transform(self, DH_table_spine: np.ndarray) -> np.ndarray:
        # Compute the transform matrix along the spine

        def t_matrix(alpha, a, theta, d):
            # Returns homogenous transform matrix for a single DH set (using the alternative DH convention)
            DH_t_matrix = np.array([[np.cos(theta),                 -1*np.sin(theta),               0,                  a],
                                    [np.sin(theta)*np.cos(alpha),   np.cos(theta)*np.cos(alpha),    -1*np.sin(alpha),   -1*d*np.sin(alpha)],
                                    [np.sin(theta)*np.sin(alpha),   np.cos(theta)*np.sin(alpha),    np.cos(alpha),      d*np.cos(alpha)],
                                    [0,                             0,                              0,                  1]])
            return DH_t_matrix

        # Full transformation matrix from spine start, to spine end
        T_total = np.identity(4)

        # Tracks intermediate spine points- useful for plotting and visualization
        interim_points = np.zeros((11,3))
        for i in range(11):
            T_total = np.dot(T_total, t_matrix(DH_table_spine[i,0], DH_table_spine[i,1], DH_table_spine[i,2], DH_table_spine[i,3]))
            # interim_points[i] = T_total[:-1,3] + self.geometric_param['com_spine']

        return T_total

    def compute_dh_table(self, angle: float) -> np.ndarray:
        # INPUT: angle as float (describes rot angle on spine)
        # thetas contains the 8 control variables for the rear spine. In radians
        dr = np.pi/180
        # Zero point offset thetas
        th0_d = np.array([90.0, -10.7, 0.0, -7.0, 0.0, -6.9, 0.0, 9.7, 0.0, 15.0, -90.0])
        th0_r = dr*th0_d
        # Control thetas. Note: th_r[2] = th_r[4] = th_r[6] = th_r[8]
        th_r = np.zeros((11,))
        th_r[2] = th_r[4] = th_r[6] = th_r[8] = angle
        # Length values for the spine
        l1 = self.spine_lengths[0]
        l2 = self.spine_lengths[1]
        l3 = self.spine_lengths[2]
        l4 = self.spine_lengths[3]
        # DH table for the spine (with open variables inside) - Size 11,4
        # Using modified DH convention - see https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
        # Scheme: alpha(n-1), a(n-1), theta(n), d(n)
        DH_table_spine = np.array([[ 0.0,    0.0, th0_r[0] + th_r[0], 0.0],
                                [ np.pi/2, l1, th0_r[1] + th_r[1], 0.0],
                                [-np.pi/2, l2, th0_r[2] + th_r[2], 0.0],
                                [ np.pi/2, l2, th0_r[3] + th_r[3], 0.0],
                                [-np.pi/2, l2, th0_r[4] + th_r[4], 0.0],
                                [ np.pi/2, l2, th0_r[5] + th_r[5], 0.0],
                                [-np.pi/2, l2, th0_r[6] + th_r[6], 0.0],
                                [ np.pi/2, l2, th0_r[7] + th_r[7], 0.0],
                                [-np.pi/2, l3, th0_r[8] + th_r[8], 0.0],
                                [ np.pi/2, l4, th0_r[9] + th_r[9], 0.0],
                                [-np.pi/2, 0.0, th0_r[10] + th_r[10], 0.0]])
        return DH_table_spine

    def servo_to_spine_angle(self, servo_angle: float, pulley_rad=0.0065) -> float:
        # Geometric properties of the spine joint (if discretized)
        l1 = 3/1000
        d1 = (4.75+0.8+0.7)/1000
        alpha = np.arctan(d1/(l1/2))
        # Change in tendon length per joint driven by pulley
        tendon_delta = (servo_angle*pulley_rad)/4
        # Tendon length inside a single joint
        lr = l1 + tendon_delta
        denom = (l1/2)**2 + d1**2
        spine_angle = np.pi - 2*alpha - np.arccos(1 - lr**2/(2*denom))
        return spine_angle

    def spine_to_servo_angle(self, spine_angle: float, pulley_rad=0.0065) -> float:
        # Convert from a spine_angle to the necessary servo angle
        l1 = 3/1000
        d1 = (4.75+0.8+0.7)/1000
        alpha = np.arctan(d1/(l1/2))
        denom = (l1/2)**2 + d1**2
        lr = np.sqrt(2*(1-np.cos(np.pi-2*alpha-spine_angle))*denom)
        servo_angle = 4*(lr-l1)/pulley_rad
        return servo_angle

    def max_angle_deflection(self, Mt: float,cd=1.2) -> float:
        # Based on a simple dynamic model of the spine
        # Each joint modelled as a discrete radial spring
        return abs(3.85*Mt/(4*cd))

    def spine_stability_forward_pass(self, theta_s: float, coords: np.ndarray) -> float:
        # Coordinates extracted
        # Coordinates are as follows
        # Front foot placement, front shoulder, (in the coordinate frame relative to spine end) rear foot placement, rear hip
        x_f, y_f, z_f = coords[0], coords[1], coords[2]
        x_s, y_s, z_s = coords[3], coords[4], coords[5]
        x_r, y_r, z_r = coords[6], coords[7], coords[8]
        x_h, y_h, z_h = coords[9], coords[10], coords[11]

        # Exported matlab equation for generalized distance
        d_com = -(1.0*((x_f + x_s)*(0.9792121*y_h + 0.9792121*y_r - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904) + (y_f + y_s)*(0.006583802*x_h + 0.006583802*x_r + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r)) + (0.00293772*np.sin(2.0*theta_s) + 0.007693575*np.sin(4.0*theta_s) + 0.003815623*np.sin(3.0*theta_s) + 0.003036031*np.sin(theta_s))*(0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904) - 1.0*(0.01429818*np.cos(theta_s)**3 - 0.05533735*np.cos(theta_s)**2 - 0.007788605*np.cos(theta_s) + 0.06047844*np.cos(theta_s)**4 + 0.01356138)*(x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r))))/((x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r))**2 + (0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904)**2)**(1/2)
        return d_com

    def spine_stability_backward_pass(self, theta_s: float, coords: np.ndarray, vel_d: float, step=0.015) -> float:
        # Extracted coordinates
        x_f, y_f, z_f = coords[0], coords[1], coords[2]
        x_s, y_s, z_s = coords[3], coords[4], coords[5]
        x_r, y_r, z_r = coords[6], coords[7], coords[8]
        x_h, y_h, z_h = coords[9], coords[10], coords[11]

        # Exported partial derivative of d_com w.r.t theta_s from Matlab
        # Yes, very nice equation
        K = (1.0*(1.0*(0.06047844*np.cos(theta_s)**4 + 0.01429818*np.cos(theta_s)**3 - 0.05533735*np.cos(theta_s)**2 - 0.007788605*np.cos(theta_s) + 0.01356138)*(0.02794953*np.cos(2.0*theta_s) + 0.005519189*np.cos(4.0*theta_s) + 0.04786366*np.cos(3.0*theta_s) + 0.01405923*np.cos(theta_s) + 0.07488884*np.cos(2.0*theta_s)*(y_h + y_r) + 3.807948*np.cos(4.0*theta_s)*(y_h + y_r) + 0.226767*np.cos(2.0*theta_s)*(z_h + z_r) - 1.020337*np.cos(4.0*theta_s)*(z_h + z_r) + 0.04202879*np.sin(2.0*theta_s)*(x_h + x_r) + 3.942278*np.sin(4.0*theta_s)*(x_h + x_r) - 0.121708*np.cos(3.0*theta_s)*(y_h + y_r) - 0.4868493*np.cos(3.0*theta_s)*(z_h + z_r) + 0.008445003*np.sin(3.0*theta_s)*(x_h + x_r) + 0.04772199*np.cos(theta_s)*(y_h + y_r) + 0.1106561*np.cos(theta_s)*(z_h + z_r) - 0.002815001*np.sin(theta_s)*(x_h + x_r)) - (0.00587544*np.cos(2.0*theta_s) + 0.0307743*np.cos(4.0*theta_s) + 0.01144687*np.cos(3.0*theta_s) + 0.003036031*np.cos(theta_s))*(0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904) + 1.0*(- 0.2419138*np.sin(theta_s)*np.cos(theta_s)**3 - 0.04289455*np.sin(theta_s)*np.cos(theta_s)**2 + 0.1106747*np.sin(theta_s)*np.cos(theta_s) + 0.007788605*np.sin(theta_s))*(x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r)) + (0.00293772*np.sin(2.0*theta_s) + 0.007693575*np.sin(4.0*theta_s) + 0.003815623*np.sin(3.0*theta_s) + 0.003036031*np.sin(theta_s))*(0.03162022*np.cos(theta_s)*np.sin(theta_s) - 0.03389211*np.sin(theta_s) + 3.891345*np.cos(2.0*theta_s)*(x_h + x_r) + 0.1877497*np.cos(theta_s)**2*np.sin(theta_s) + 0.0433858*np.cos(theta_s)**3*np.sin(theta_s) + 0.04523312*np.cos(theta_s)*(x_h + x_r) + 0.2480776*np.sin(theta_s)*(y_h + y_r) + 0.5676143*np.sin(theta_s)*(z_h + z_r) - 7.747465*np.cos(theta_s)**4*(x_h + x_r) + 23.24239*np.cos(theta_s)**2*np.sin(theta_s)**2*(x_h + x_r) - 14.92965*np.cos(theta_s)*np.sin(theta_s)*(y_h + y_r) + 4.501203*np.cos(theta_s)*np.sin(theta_s)*(z_h + z_r) - 0.235415*np.cos(theta_s)*np.sin(theta_s)**2*(x_h + x_r) - 0.7378223*np.cos(theta_s)**2*np.sin(theta_s)*(y_h + y_r) + 29.93391*np.cos(theta_s)**3*np.sin(theta_s)*(y_h + y_r) - 1.844017*np.cos(theta_s)**2*np.sin(theta_s)*(z_h + z_r) - 8.020766*np.cos(theta_s)**3*np.sin(theta_s)*(z_h + z_r)) + (x_f + x_s)*(0.03162022*np.cos(theta_s)*np.sin(theta_s) - 0.03389211*np.sin(theta_s) + 3.891345*np.cos(2.0*theta_s)*(x_h + x_r) + 0.1877497*np.cos(theta_s)**2*np.sin(theta_s) + 0.0433858*np.cos(theta_s)**3*np.sin(theta_s) + 0.04523312*np.cos(theta_s)*(x_h + x_r) + 0.2480776*np.sin(theta_s)*(y_h + y_r) + 0.5676143*np.sin(theta_s)*(z_h + z_r) - 7.747465*np.cos(theta_s)**4*(x_h + x_r) + 23.24239*np.cos(theta_s)**2*np.sin(theta_s)**2*(x_h + x_r) - 14.92965*np.cos(theta_s)*np.sin(theta_s)*(y_h + y_r) + 4.501203*np.cos(theta_s)*np.sin(theta_s)*(z_h + z_r) - 0.235415*np.cos(theta_s)*np.sin(theta_s)**2*(x_h + x_r) - 0.7378223*np.cos(theta_s)**2*np.sin(theta_s)*(y_h + y_r) + 29.93391*np.cos(theta_s)**3*np.sin(theta_s)*(y_h + y_r) - 1.844017*np.cos(theta_s)**2*np.sin(theta_s)*(z_h + z_r) - 8.020766*np.cos(theta_s)**3*np.sin(theta_s)*(z_h + z_r)) - (y_f + y_s)*(0.02794953*np.cos(2.0*theta_s) + 0.005519189*np.cos(4.0*theta_s) + 0.04786366*np.cos(3.0*theta_s) + 0.01405923*np.cos(theta_s) + 0.07488884*np.cos(2.0*theta_s)*(y_h + y_r) + 3.807948*np.cos(4.0*theta_s)*(y_h + y_r) + 0.226767*np.cos(2.0*theta_s)*(z_h + z_r) - 1.020337*np.cos(4.0*theta_s)*(z_h + z_r) + 0.04202879*np.sin(2.0*theta_s)*(x_h + x_r) + 3.942278*np.sin(4.0*theta_s)*(x_h + x_r) - 0.121708*np.cos(3.0*theta_s)*(y_h + y_r) - 0.4868493*np.cos(3.0*theta_s)*(z_h + z_r) + 0.008445003*np.sin(3.0*theta_s)*(x_h + x_r) + 0.04772199*np.cos(theta_s)*(y_h + y_r) + 0.1106561*np.cos(theta_s)*(z_h + z_r) - 0.002815001*np.sin(theta_s)*(x_h + x_r))))/((x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r))**2 + (0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904)**2)**(1/2) + (0.5*(2*(0.02794953*np.cos(2.0*theta_s) + 0.005519189*np.cos(4.0*theta_s) + 0.04786366*np.cos(3.0*theta_s) + 0.01405923*np.cos(theta_s) + 0.07488884*np.cos(2.0*theta_s)*(y_h + y_r) + 3.807948*np.cos(4.0*theta_s)*(y_h + y_r) + 0.226767*np.cos(2.0*theta_s)*(z_h + z_r) - 1.020337*np.cos(4.0*theta_s)*(z_h + z_r) + 0.04202879*np.sin(2.0*theta_s)*(x_h + x_r) + 3.942278*np.sin(4.0*theta_s)*(x_h + x_r) - 0.121708*np.cos(3.0*theta_s)*(y_h + y_r) - 0.4868493*np.cos(3.0*theta_s)*(z_h + z_r) + 0.008445003*np.sin(3.0*theta_s)*(x_h + x_r) + 0.04772199*np.cos(theta_s)*(y_h + y_r) + 0.1106561*np.cos(theta_s)*(z_h + z_r) - 0.002815001*np.sin(theta_s)*(x_h + x_r))*(x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r)) - 2*(0.03162022*np.cos(theta_s)*np.sin(theta_s) - 0.03389211*np.sin(theta_s) + 3.891345*np.cos(2.0*theta_s)*(x_h + x_r) + 0.1877497*np.cos(theta_s)**2*np.sin(theta_s) + 0.0433858*np.cos(theta_s)**3*np.sin(theta_s) + 0.04523312*np.cos(theta_s)*(x_h + x_r) + 0.2480776*np.sin(theta_s)*(y_h + y_r) + 0.5676143*np.sin(theta_s)*(z_h + z_r) - 7.747465*np.cos(theta_s)**4*(x_h + x_r) + 23.24239*np.cos(theta_s)**2*np.sin(theta_s)**2*(x_h + x_r) - 14.92965*np.cos(theta_s)*np.sin(theta_s)*(y_h + y_r) + 4.501203*np.cos(theta_s)*np.sin(theta_s)*(z_h + z_r) - 0.235415*np.cos(theta_s)*np.sin(theta_s)**2*(x_h + x_r) - 0.7378223*np.cos(theta_s)**2*np.sin(theta_s)*(y_h + y_r) + 29.93391*np.cos(theta_s)**3*np.sin(theta_s)*(y_h + y_r) - 1.844017*np.cos(theta_s)**2*np.sin(theta_s)*(z_h + z_r) - 8.020766*np.cos(theta_s)**3*np.sin(theta_s)*(z_h + z_r))*(0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904))*((x_f + x_s)*(0.9792121*y_h + 0.9792121*y_r - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904) + (y_f + y_s)*(0.006583802*x_h + 0.006583802*x_r + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r)) + (0.00293772*np.sin(2.0*theta_s) + 0.007693575*np.sin(4.0*theta_s) + 0.003815623*np.sin(3.0*theta_s) + 0.003036031*np.sin(theta_s))*(0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904) - 1.0*(0.06047844*np.cos(theta_s)**4 + 0.01429818*np.cos(theta_s)**3 - 0.05533735*np.cos(theta_s)**2 - 0.007788605*np.cos(theta_s) + 0.01356138)*(x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r))))/((x_f + 0.006583802*x_h + 0.006583802*x_r + x_s + 0.01397477*np.sin(2.0*theta_s) + 0.001379797*np.sin(4.0*theta_s) + 0.01595455*np.sin(3.0*theta_s) + 0.01405923*np.sin(theta_s) - 0.02101439*np.cos(2.0*theta_s)*(x_h + x_r) - 0.9855694*np.cos(4.0*theta_s)*(x_h + x_r) + 0.03744442*np.sin(2.0*theta_s)*(y_h + y_r) + 0.9519869*np.sin(4.0*theta_s)*(y_h + y_r) + 0.1133835*np.sin(2.0*theta_s)*(z_h + z_r) - 0.2550841*np.sin(4.0*theta_s)*(z_h + z_r) - 0.002815001*np.cos(3.0*theta_s)*(x_h + x_r) - 0.04056932*np.sin(3.0*theta_s)*(y_h + y_r) - 0.1622831*np.sin(3.0*theta_s)*(z_h + z_r) + 0.002815001*np.cos(theta_s)*(x_h + x_r) + 0.04772199*np.sin(theta_s)*(y_h + y_r) + 0.1106561*np.sin(theta_s)*(z_h + z_r))**2 + (0.9792121*y_h - 1.0*y_f + 0.9792121*y_r - 1.0*y_s - 0.2000972*z_h - 0.2000972*z_r - 0.03389211*np.cos(theta_s) - 1.945673*np.sin(2.0*theta_s)*(x_h + x_r) + 0.01581011*np.cos(theta_s)**2 + 0.06258325*np.cos(theta_s)**3 + 0.01084645*np.cos(theta_s)**4 + 0.2480776*np.cos(theta_s)*(y_h + y_r) + 0.5676143*np.cos(theta_s)*(z_h + z_r) - 0.04523312*np.sin(theta_s)*(x_h + x_r) - 7.464827*np.cos(theta_s)**2*(y_h + y_r) - 0.2459408*np.cos(theta_s)**3*(y_h + y_r) + 7.483476*np.cos(theta_s)**4*(y_h + y_r) + 2.250601*np.cos(theta_s)**2*(z_h + z_r) - 0.6146722*np.cos(theta_s)**3*(z_h + z_r) - 2.005191*np.cos(theta_s)**4*(z_h + z_r) + 0.07847166*np.sin(theta_s)**3*(x_h + x_r) + 7.747465*np.cos(theta_s)**3*np.sin(theta_s)*(x_h + x_r) + 0.02801904)**2)**(3/2)

        theta_s_dot = 1/K*vel_d
        theta_s = theta_s + step*theta_s_dot
        theta_s = max(min(theta_s,self.theta_s_max),self.theta_s_min)
        return theta_s
