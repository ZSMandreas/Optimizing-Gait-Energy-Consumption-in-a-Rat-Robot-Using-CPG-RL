from threading import Thread

from ps4_ctrl import joy_controller

from ToSim import SimModel
from Controller import MouseController
import time

def cmdInput(tInput):
	user_input = input('Type user input: ')
	while True:
		user_input = input('Type user input: ')
		tInput[0] = user_input
		if user_input == "q":
			break

if __name__ == '__main__':
	tlist = ["i"]
	key_input = Thread(target=cmdInput, args = (tlist,))
	key_input.start()

	max_fre = 1.0
	max_spine_A = 30
	fre = 0.5
	time_step = 0.002
	theController = MouseController(fre, time_step)

	theMouse = SimModel("../models/dynamic_4l_t3.xml")
	theMouse.initializing()

	for i in range(100):
		ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]
		theMouse.runStep(ctrlData, time_step)

	t_buttons_data = [0]*4
	t_rocker_val = [0]*4
	t_arrow_key = [0]*2

	done = False
	while not done:
		if tlist[0] == "q":
			done = True
		if tlist[0] == "w":
			t_rocker_val[1] = -1.0
		elif tlist[0] == "s":
			t_rocker_val[1] = 1.0
		elif tlist[0] == "a":
			t_rocker_val[2] = -1.0
		elif tlist[0] == "d":
			t_rocker_val[2] = 1.0
		elif tlist[0] == "x":
			t_rocker_val[2] = 0.0
		elif tlist[0] == "e":
			t_rocker_val[1] = 0.0
		#t_rocker_val[1] = -1.0
		#t_rocker_val[2] = 0.0
		#t_arrow_key[0] = 0.0
		fre_g = 0 - t_rocker_val[1]
		spine_g = -1* t_rocker_val[2]
		head_g = t_arrow_key[0]
		theController.update_motion(fre_g, spine_g, head_g)
		tCtrlData = theController.runStep()
		ctrlData = tCtrlData
		theMouse.runStep(ctrlData, time_step)
		#print(tlist)


	print('The end')