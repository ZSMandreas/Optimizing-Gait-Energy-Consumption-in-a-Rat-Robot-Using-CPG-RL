import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import math
from forleg_angle import ForeLegM

if __name__ == '__main__':
	# 1: AE 	2: AB
	# 3: BC 	4: CD
	# 5: DE 	6: EF (DE+EF ~= DF)
	#leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
	leg_params = [0.026, 0.0128, 0.0118, 0.040, 0.015, 0.040, 0.0205, 73*np.pi/180]

	#'l4': 0.0205,'alpha':73*np.pi/180

	fl_left = ForeLegM(leg_params)
	
	PI = math.pi
	stepNum = 100
	initRad = PI*3/4
	stepRad = (2*initRad) / stepNum
	
	points_list = []
	#q1 = 0
	#for i in range(int(stepNum/2)):
	q1 = initRad
	#cur_points = fl_left.angel_2_pos(0, 0)
	cur_points = fl_left.angel_2_pos(math.pi*0/180, 0)


	plt.xlabel(r'$Y\ data$')
	plt.ylabel(r'$Z\ data$')
	plt.gca().set_aspect('equal')
	plt.xlim(-0.1, 0.1)
	plt.ylim(-0.1, 0.1)
	plt.xticks(np.arange(-0.1, 0.1, step=0.02)) 
	plt.yticks(np.arange(-0.1, 0.1, step=0.02))    
	plt.grid()

	lines = plt.plot([], [], 'r-',  # EA
			[], [], 'k-',			# AB
			[], [], 'r-',			# BC
			[], [], 'b-',			# CD
			[], [], 'b-',			# DE
			[], [], 'b-',			# EF
			[], [], 'b-',)			# FG

	print(cur_points)
	ty = [cur_points[4][0]]
	tz = [cur_points[4][1]]
	for ti in range(7):
		ty.append(cur_points[ti][0])
		tz.append(cur_points[ti][1])

	for i in range(7):
		lines[i].set_data([ty[i], ty[i+1]], [tz[i], tz[i+1]])

	plt.show()
