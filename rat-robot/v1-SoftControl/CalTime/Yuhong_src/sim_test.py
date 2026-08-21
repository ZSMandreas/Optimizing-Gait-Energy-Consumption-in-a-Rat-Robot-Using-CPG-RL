from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time


# --------------------
RUN_STEPS = 10000 #320
if __name__ == '__main__':
	theMouse = SimModel("../models/dynamic_4l_t3.xml")

	fre = 1.00
	theController = MouseController(fre)
	for i in range(500):
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
	print("Traj --> ", theController.traj_time)
	print("Spine --> ", theController.sVal_time)
	print("Leg --> ", theController.qVal_time)
	total_t = theController.traj_time + theController.sVal_time + \
		theController.qVal_time
	print('Others --> ', timeCost - total_t)

	dis = theMouse.drawPath()
	print("py_v --> ", dis/timeCost)
	print("sim_v --> ", dis/(RUN_STEPS*0.002))