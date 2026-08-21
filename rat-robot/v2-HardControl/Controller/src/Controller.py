import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel

class MouseController(object):
	"""docstring for MouseController"""
	def __init__(self, fre, time_step, spine_angle):
		super(MouseController, self).__init__()
		PI = np.pi
		self.curRad = 0
		
		self.turn_F = 0*PI/180
		self.turn_H = 12*PI/180
		self.pathStore = LegPath()
		
		self.phaseDiff = [0, PI, PI, 0]			# Trot
		self.period = 2/2
		self.fre_cyc = fre
		self.time_step = time_step
		
		self.SteNum = int(1/(time_step*self.fre_cyc))
		print("----> ", self.SteNum)
		self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.spine_A =2*spine_angle#10 a_s = 2theta_s
		print("angle --> ", spine_angle)#self.spine_A)
		self.spine_A = self.spine_A*PI/180
		# --------------------------------------------------------------------- #
		leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
		self.fl_left = LegModel(leg_params)
		self.fl_right = LegModel(leg_params)
		self.hl_left = LegModel(leg_params)
		self.hl_right = LegModel(leg_params)
		# --------------------------------------------------------------------- #
		self.stepDiff = [0,0,0,0]
		for i in range(4):
			self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i]/(2*PI))
		self.stepDiff.append(int(self.SteNum * self.spinePhase/(2*PI)))
		self.trgXList = [[],[],[],[]]
		self.trgYList = [[],[],[],[]]

	def getLegCtrl(self, leg_M, curRad, leg_ID):
		turnAngle = self.turn_F
		leg_flag = "F"
		if leg_ID > 1:
			leg_flag = "H"
			turnAngle = self.turn_H

		currentPos = self.pathStore.getOvalPathPoint(curRad, leg_flag, self.period)
		trg_x = currentPos[0]
		trg_y = currentPos[1]
		self.trgXList[leg_ID].append(trg_x)
		self.trgYList[leg_ID].append(trg_y)

		tX = math.cos(turnAngle)*trg_x - math.sin(turnAngle)*trg_y;
		tY = math.cos(turnAngle)*trg_y + math.sin(turnAngle)*trg_x;
		qVal = leg_M.pos_2_angle(tX, tY)
		return qVal

	def getSpineVal(self, curRad):
		return self.spine_A*math.cos(curRad-self.spinePhase)

	def reset(self):
		self.fre_cyc = 0
		self.curRad = 0
		self.SteNum = 0
		self.spine_A = 0
		self.head_A = 0

	def update_motion(self, max_fre, fre_g, spine_g, head_g):
		self.fre_cyc = max_fre * fre_g
		self.spine_A = spine_g * 30 * np.pi/180
		self.head_A = head_g * 15 * np.pi/180
		self.SteNum = 0
		if self.fre_cyc != 0:
			self.SteNum = int(1/(self.time_step*self.fre_cyc))

	def runStep(self):
		foreLeg_left_q = self.getLegCtrl(self.fl_left, 
			self.curRad+ self.phaseDiff[0], 0)
		foreLeg_right_q = self.getLegCtrl(self.fl_right, 
			self.curRad + self.phaseDiff[1], 1)
		hindLeg_left_q = self.getLegCtrl(self.hl_left, 
			self.curRad + self.phaseDiff[2], 2)
		hindLeg_right_q = self.getLegCtrl(self.hl_right, 
			self.curRad + self.phaseDiff[3], 3)

		spineRad = self.curRad
		spine = self.spine_A

		step_rad = 0
		if self.SteNum != 0:
			step_rad = 2*np.pi/self.SteNum
		self.curRad += step_rad
		if self.curRad > 2*np.pi:
			self.curRad -= 2*np.pi

		ctrlData = []


		#foreLeg_left_q = [1,0]
		#foreLeg_right_q = [1,0]
		#hindLeg_left_q = [-1,0]
		#hindLeg_right_q = [-1,0]
		ctrlData.extend(foreLeg_left_q)
		ctrlData.extend(foreLeg_right_q)
		ctrlData.extend(hindLeg_left_q)
		ctrlData.extend(hindLeg_right_q)
		ctrlData.append(0)				# Tail
		ctrlData.append(self.head_A)	# Head
		ctrlData.append(0)
		ctrlData.append(spine)
		return ctrlData
		