from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time

from ps4_ctrl import joy_controller


if __name__ == '__main__':
	max_fre = 1.0
	max_spine_A = 30

	fre = 0.5
	time_step = 0.002
	init_spine_angle = 0

	theMouse = SimModel("../models/dynamic_4l.xml")
	theController = MouseController(fre, time_step, init_spine_angle)

	init_ctrl = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
	for i in range(100):
		ctrlData = init_ctrl
		theMouse.runStep(ctrlData, time_step)
	theMouse.initializing()

	t_buttons_data = [0]*4
	t_rocker_val = [0]*4
	t_arrow_key = [0]*2

	the_joy = joy_controller()
	done = False

	while not done:
		t_buttons_data, t_rocker_val, t_arrow_key = the_joy.runStep()
		if t_buttons_data[0] == 1:
			done = True
		elif t_buttons_data[2] == 1:
			theController.reset()
			the_joy.reset()
			t_buttons_data = [0]*4
			t_rocker_val = [0]*4
			t_arrow_key = [0]*2
			ctrlData = init_ctrl
			theMouse.runStep(ctrlData, time_step)
			continue
		if t_rocker_val[1] == 0 and t_rocker_val[2] == 0:
			ctrlData = init_ctrl
			theMouse.runStep(ctrlData, time_step)
			continue
		fre_g = 0 - t_rocker_val[1]
		spine_g = -1* t_rocker_val[2]
		head_g = t_arrow_key[0]
		theController.update_motion(max_fre, fre_g, spine_g, head_g)
		tCtrlData = theController.runStep()
		ctrlData = tCtrlData
		theMouse.runStep(ctrlData, time_step)

	print('The end')