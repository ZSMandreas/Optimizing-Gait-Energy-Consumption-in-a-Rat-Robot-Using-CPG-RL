from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

class SimModel(object):
	"""docstring for SimModel"""
	def __init__(self, modelPath):
		super(SimModel, self).__init__()
		self.model = load_model_from_path(modelPath)
		self.sim = MjSim(self.model)
		self.viewer = MjViewer(self.sim)

		self.viewer.cam.lookat[0] += 0.1
		self.viewer.cam.lookat[1] += 0
		self.viewer.cam.lookat[2] += 0
		self.viewer.cam.elevation = 0
		self.viewer.cam.azimuth = 180
		#self.viewer.cam.lookat[0] += 0.5
		#self.viewer.cam.lookat[1] += 0.5
		#self.viewer.cam.elevation = 0
		#.viewer.cam.azimuth = 0
		self.viewer.cam.distance = self.model.stat.extent*0.05

		self.sim_state = self.sim.get_state()
		self.sim.set_state(self.sim_state)

		self.legRealPoint_y = []
		self.legRealPoint_z = []

	def runStep(self, ctrlData, cur_time_step):
		# ------------------------------------------ #
		# ID 0, 1 left-fore leg and coil 
		# ------------------------------------------ #
		step_num = int(cur_time_step/0.002)
		self.sim.data.ctrl[:] = ctrlData
		for i in range(step_num):
			self.sim.step()
			self.viewer.render()
		check_point_1 = self.sim.data.get_site_xpos('thigh_up_point_1')
		check_point_2 = self.sim.data.get_site_xpos('thigh_up_point_2')

		dis_x = check_point_1[0] - check_point_2[0] 
		dis_y = check_point_1[1] - check_point_2[1] 
		dis_z = check_point_1[2] - check_point_2[2] 
		dis_w = math.sqrt(dis_x*dis_x + dis_y*dis_y + dis_z*dis_z)
		'''
		print(dis_w)
		if dis_w > 0.0015:
			print('\033[1;31m [ERROR] \033[0m！ Leg Connector breaks connection!')
			sys.exit(-1)
			'''
		
		originPoint = self.sim.data.get_site_xpos('leg_link')
		currentPoint = self.sim.data.get_site_xpos('ankle')
		tY = currentPoint[1]-originPoint[1]
		tZ = currentPoint[2]-originPoint[2]
		self.legRealPoint_y.append(tY)
		self.legRealPoint_z.append(tZ)
		
	def getTime(self):
		return self.sim.data.time
