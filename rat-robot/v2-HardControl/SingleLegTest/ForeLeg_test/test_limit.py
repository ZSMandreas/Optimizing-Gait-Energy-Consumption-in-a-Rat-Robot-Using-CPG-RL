import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import math
#from foreLeg_ideal import ForeLegM
from foreLeg_limit import ForeLegM

if __name__ == '__main__':
	# 1: AE 	2: AB
	# 3: BC 	4: CD
	# 5: DE 	6: EF (DE+EF ~= DF)
	leg_params = [0.045059, 0.0088, 0.012022, 0.056223, 0.015811, 0.050048]

	fl_left = ForeLegM(leg_params)
	
	PI = math.pi
	stepNum = 100
	initRad = PI*3/4
	stepRad = (2*initRad) / stepNum
	
	points_list = []
	#q1 = 0
	#for i in range(int(stepNum/2)):
	q1 = initRad
	for i in range(stepNum):
		init_q2 = q1 - PI*2/4
		q2 = init_q2
		step_points = []
		while q2 < PI*5/6:
			#print([q1, q2])
			cur_points = fl_left.angel_2_pos(q1, q2)
			if len(cur_points) == 0:
				break
			step_points.append(cur_points)
			q2 += stepRad

		q2 = init_q2
		while q2 > -PI*5/6:
			cur_points = fl_left.angel_2_pos(q1, q2)
			if len(cur_points) == 0:
				break
			step_points.append(cur_points)
			q2 -= stepRad

		points_list.append(step_points)
		q1 -= stepRad

	fig, ax = plt.subplots()
	leg_list = []
	for i in range(stepNum):
		pL = len(points_list[i])
		#print(pL)
		for j in range(pL):
			leg_list.append(points_list[i][j])
		y_list = [points_list[i][j][-1][0] for j in range(pL)]
		z_list = [points_list[i][j][-1][1] for j in range(pL)]
		l1 = ax.scatter(y_list, z_list, color='lightgreen')
	ax.set_xlabel(r'$Y\ data$')
	ax.set_ylabel(r'$Z\ data$')
	ax.set_aspect(1)
	ax.set_xlim(-0.1, 0.1)
	ax.set_ylim(-0.1, 0.1)
	ax.xaxis.set_ticks(np.arange(-0.1, 0.1, step=0.02)) 
	ax.yaxis.set_ticks(np.arange(-0.1, 0.1, step=0.02))    
	ax.grid()

	lines = plt.plot([], [], 'r-',  # EA
			[], [], 'k-',			# AB
			[], [], 'r-',			# BC
			[], [], 'b-',			# CD
			[], [], 'b-',			# DE
			[], [], 'b-',)			# EF

	def update(k):
		ty = [leg_list[k][4][0]]
		tz = [leg_list[k][4][1]]
		for ti in range(6):
			ty.append(leg_list[k][ti][0])
			tz.append(leg_list[k][ti][1])

		for i in range(6):
			lines[i].set_data([ty[i], ty[i+1]], [tz[i], tz[i+1]])
			#lines.append(line)
		'''
		EA, = ax.plot([ty[0], ty[1]], [tz[0], tz[1]], 'red', lw=2)
		AB, = ax.plot([ty[1], ty[2]], [tz[1], tz[2]], 'black', lw=2)
		BC, = ax.plot([ty[2], ty[3]], [tz[2], tz[3]], 'red', lw=2)
		CD, = ax.plot([ty[3], ty[4]], [tz[3], tz[4]], 'g', lw=2)
		DE, = ax.plot([ty[4], ty[5]], [tz[4], tz[5]], 'g', lw=2)
		EF, = ax.plot([ty[5], ty[6]], [tz[5], tz[6]], 'g', lw=2)
		'''
		#line, = ax.plot(ty, tz, 'g--', lw=2)
		return lines

	ani = animation.FuncAnimation(fig, update, frames=len(leg_list), interval=100,)
	#ani.save('ideal.gif', writer='imagemagick')

	plt.show()
