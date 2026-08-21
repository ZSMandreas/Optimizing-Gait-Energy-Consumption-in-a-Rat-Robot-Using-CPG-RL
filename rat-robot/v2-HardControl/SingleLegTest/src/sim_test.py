from ToSim import SimModel
from Controller import LegController
import matplotlib.pyplot as plt
import time

INIT_STEPS = 1000
SIM_STEPS = 5000
if __name__ == '__main__':
	theLeg = SimModel("../Model/fore_right_leg.xml")

	fre = 0.5
	time_step = 0.002
	theController = LegController(fre, time_step)

	for i in range(INIT_STEPS):
		ctrlData = [0,0]
		theLeg.runStep(ctrlData, time_step)
	#print(theLeg.legRealPoint_y[-1], theLeg.legRealPoint_z[-1])

	for i in range(SIM_STEPS):
		tCtrlData = theController.runStep()
		ctrlData = tCtrlData
		theLeg.runStep(ctrlData, time_step)

	#'''
	plt.plot(theController.trgList_y, theController.trgList_z)
	plt.plot(theLeg.legRealPoint_y, theLeg.legRealPoint_z)
	
	plt.show()