import argparse

from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time


RUN_STEPS = 10000
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Description.")
	parser.add_argument('--fre', default=1.25,
		type=float, help="Gait stride")
	args = parser.parse_args()

	theMouse = SimModel("../models/dynamic_4l_t3.xml")

	theController = MouseController(args.fre)
	for i in range(500):
		#ctrlData = 0
		ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
		theMouse.runStep(ctrlData)
	theMouse.initializing()
	start = time.time()
	for i in range(RUN_STEPS):
		#print("Step --> ", i)
		tCtrlData = theController.runStep()				# No Spine
		#tCtrlData = theController.runStep_spine()		# With Spine
		ctrlData = tCtrlData
		theMouse.runStep(ctrlData)
	end = time.time()
	timeCost = end-start
	print("Time -> ", timeCost)

	#'''
	fig, axs = plt.subplots(2,2)
	subTitle = ["Fore Left Leg", "Fore Right Leg",
		"Hind Left Leg", "Hind Right Leg"]
	for i in range(4):
		pos_1 = int(i/2)
		pos_2 = int(i%2)
		print(pos_1, pos_2)
		axs[pos_1,pos_2].set_title(subTitle[i])
		axs[pos_1,pos_2].plot(theController.trgXList[i][500:], theController.trgYList[i][500:]
			, label='Target trajectory')
		axs[pos_1,pos_2].plot(theMouse.legRealPoint_x[i][500:], theMouse.legRealPoint_y[i][500:]
			, label='Simulation trajectory')
	fig.suptitle('Control trajectory of model from Yuhong', fontsize=15)
	plt.subplots_adjust(hspace=0.5,wspace=0.3)
	plt.savefig('new_traj_125.png')
	plt.show()
