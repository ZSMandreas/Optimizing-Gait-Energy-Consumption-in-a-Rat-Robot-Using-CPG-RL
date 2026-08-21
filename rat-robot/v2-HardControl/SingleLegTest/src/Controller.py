import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel

class LegController(object):
	"""docstring for MouseController"""
	def __init__(self, fre, time_step):
		super(LegController, self).__init__()
		PI = np.pi
		self.curRad = 0
		self.pathStore = LegPath()
		# --------------------------------------------------------------------- #
		#self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
		#self.period = 3/2
		#self.SteNum = 36							#32 # Devide 2*PI to multiple steps
		#self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.period = 2/2
		self.fre_cyc = fre
		self.time_step = time_step
		self.SteNum = int(1/(self.time_step*self.fre_cyc))
		print("----> ", self.SteNum)
		# --------------------------------------------------------------------- #
		# 0: AE 	1: AB 	2: BC
		# 3: CD 	4: DE 	5: EF (DE+EF ~= DF)
		leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
		self.fl_left = LegModel(leg_params)

		self.trgList_y = []
		self.trgList_z = []

	def getLegCtrl(self, leg_M, curRad, leg_ID):
		#curStep = curStep % self.SteNum
		leg_flag = "F"
		if leg_ID > 1:
			leg_flag = "H"

		currentPos = self.pathStore.getOvalPathPoint(curRad, leg_flag, self.period)
		trg_y = currentPos[0]
		trg_z = currentPos[1]
		self.trgList_y.append(trg_y)
		self.trgList_z.append(trg_z)

		qVal = leg_M.pos_2_angle(trg_y, trg_z)
		return qVal

	
	def runStep(self):
		foreLeg_left_q = self.getLegCtrl(self.fl_left, 
			self.curRad, 0)

		step_rad = 0
		if self.SteNum != 0:
			step_rad = 2*np.pi/self.SteNum
		self.curRad += step_rad
		if self.curRad > 2*np.pi:
			self.curRad -= 2*np.pi

		return foreLeg_left_q
		