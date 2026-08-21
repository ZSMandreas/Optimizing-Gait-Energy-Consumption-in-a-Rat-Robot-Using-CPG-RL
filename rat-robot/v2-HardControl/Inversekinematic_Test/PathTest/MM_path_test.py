import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


from Controller import LegController
import matplotlib.pyplot as plt
import time

INIT_STEPS = 100
SIM_STEPS = 500
if __name__ == '__main__':

	fre = 0.5
	time_step = 0.01
	theController = LegController(fre, time_step)

	init_pos = theController.fl_left.angel_2_pos(0, 0)
	print(init_pos)


	points_list = []
	for i in range(SIM_STEPS):
		ctrlData = theController.runStep()
		
		cur_points = theController.fl_left.angel_2_pos(ctrlData[0], ctrlData[1])
		#print(cur_points[-1], ctrlData)
		points_list.append(cur_points)


	fig, ax = plt.subplots()
	leg_list = []
	pL = len(points_list)
	#print(pL)
	for j in range(pL):
		leg_list.append(points_list[j])
	y_list = [points_list[j][-1][0] for j in range(pL)]
	z_list = [points_list[j][-1][1] for j in range(pL)]
	#l1 = ax.scatter(y_list, z_list, color='lightgreen')
	l1 = ax.plot(y_list, z_list, color = '#a5532a', linewidth=1, linestyle='--')
	
	ax.set_xlabel(r'$Y\ data$')
	ax.set_ylabel(r'$Z\ data$')
	ax.set_aspect(1)
	ax.set_xlim(-0.05, 0.05)
	ax.set_ylim(-0.075, 0.05)
	ax.xaxis.set_ticks(np.arange(-0.05, 0.05, step=0.02)) 
	ax.yaxis.set_ticks(np.arange(-0.075, 0.05, step=0.02))    
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
		return lines

	ani = animation.FuncAnimation(fig, update, frames=len(leg_list), interval=10,)
	plt.show()