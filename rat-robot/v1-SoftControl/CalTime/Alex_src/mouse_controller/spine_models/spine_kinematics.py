import numpy as np


class Spine_Kinematics:

    def __init__(self, geoemtric_param):
        print("Initializing spine model")
        self.spine_angle = 0.0
        self.geometric_param = geoemtric_param
        self.helpers = Spine_Helpers()

    def compute_spine_transform(self, servo_angle, leg_positions):
        # 
        spine_angle = self.servo_to_spine_angle(servo_angle)
        DH_params = self.compute_dh_table(spine_angle)
        T_total, interim_points, rear_block = self.compute_transform(DH_params, leg_positions)
        
    def servo_to_spine_angle(self, servo_angle, pulley_rad=0.0065):
        # Geometric properties of the spine joint
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

    def max_angle_deflection(self, Mt,cd=1.2):

        return abs(3.85*Mt/(4*cd))

    def compute_transform(self, DH_table_spine, leg_positions):
        # Leg positions: FL, FR, RL, RR
        # 

        leg_pos_rl = leg_positions[3,:] + self.geometric_param['se_rl']
        leg_pos_rr = leg_positions[4,:] + self.geometric_param['se_rr']
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
            interim_points[i] = T_total[:-1,3] + self.geometric_param['com_spine']

        rear_block = np.ones((4,6))

        # Contains the most important elements of the rear leg
        # Rear COM, rear hip (left & right), rear footend (left and right), rear spine_end
        rear_block[:-1,:] = (np.array([self.geometric_param['se_comr'],
                                self.geometric_param['se_rl'],
                                self.geometric_param['se_rr'],
                                leg_pos_rl,
                                leg_pos_rr,
                                leg_positions[4,:],
                                self.geometric_param['se']])).T

        com_spine_pos = np.zeros((4,))
        com_spine_pos[:-1] = self.geometric_param['com_spine']
        # This operation shifts all elements to be relative within the whole mouse model
        rear_block_zeroed = ((np.dot(T_total,rear_block)).T + com_spine_pos)[:,:-1]
        rear_block_plot = np.array([rear_block_zeroed[3],
                                    rear_block_zeroed[0],
                                    rear_block_zeroed[1],
                                    rear_block_zeroed[2],
                                    rear_block_zeroed[0]])

        return (T_total, interim_points, rear_block_zeroed)

    def compute_dh_table(self, angle: float):
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
        l1 = 0.0016
        l2 = 0.0070
        l3 = 0.0092
        l4 = 0.0014
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

    def spine_stride_extension(self, timing,vel, offset=0, scaling=0.4):
        # THis function helps extend the spine stride during gait.
        # Timing value: is normalized time value
        scale = min(np.abs(vel)*1.5, scaling)
        q_spine = scale*np.cos(2*np.pi*timing+offset)
        return q_spine

class Spine_Helpers:
    # Special functions to support the velocity direction of the spine

    def __init__(self):
        print("Spine helpers")

    def balance_line(self, leg_positions, contact_params, com_spine):
        # Leg positions after going through spine transform
        # FL, FR, RL, RR

        leg1, leg2 = self.return_which_diagonals(leg_positions, contact_params)
        if type(leg1) == int:
            # If not diagonals then return no velocity
            return np.array([0,0])
        
        line_params = self.compute_line_params(leg1, leg2)
        A = line_params[0]/line_params[1]

        distance = ((A*com_spine[0] + line_params[2]*com_spine[1] + line_params[3])
                    /(np.sqrt(A**2 + line_params[2]**2)))
        direction_point = np.sign(distance)


        # Check which side of stability line the COM is on to decide which way to push it
        # Relevant for the stability model
        if (direction_point >= 0):
            dist = np.abs(distance)
            com_vel = dist*np.array([-1*line_params[0],line_params[1]])
        else:
            dist = np.abs(distance)
            com_vel = dist*np.array([line_params[0],-1*line_params[1]])

        return com_vel

    def compute_line_params(self, leg1, leg2):
        # Returns line parameters for a 2D line
        # Gradient A (split into its two components)
        # B (coefficient of Y), C (y-intercept)
        dy = leg1[1] - leg2[1]
        dx = leg1[0] - leg2[0]
        B = -1
        C = leg1[1] - (dy/dx)*leg1[0]
        return np.array([dy,dx,B,C])


    def return_which_diagonals(self, leg_positions, contact_params):
        # Leg positions (4,3)
        # contact_params (4,)

        if (contact_params[0] == 1 and contact_params[3] == 1):
            return (leg_positions[0], leg_positions[3])
        elif (contact_params[1] == 1 and contact_params[2] == 1):
            return (leg_positions[1], leg_positions[2])
        else:
            return (-1, -1)
