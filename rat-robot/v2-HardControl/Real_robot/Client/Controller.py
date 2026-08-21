import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel

class MouseController(object):
	"""docstring for MouseController"""
	def __init__(self, timeStep, fre):
		super(MouseController, self).__init__()
		PI = np.pi
		self.curStep = 0	# Spine
			
		#self.turn_F = 0#-2*PI/180
		#self.turn_H = 0#5*PI/180
		self.turn_F = 0*PI/180
		self.turn_H = 0*PI/180
		self.pathStore = LegPath()
		# [LF, RF, LH, RH]
		# --------------------------------------------------------------------- #
		'''
		self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
		#self.phaseDiff = [0, PI, PI*3/2, PI*1/2]	# Walk
		self.period = 3/2
		self.SteNum = 64							#32 # Devide 2*PI to multiple steps
		self.spinePhase = self.phaseDiff[3]
		'''
		# --------------------------------------------------------------------- #
		self.phaseDiff = [0, PI, PI, 0]			# Trot
		#self.phaseDiff = [PI, 0, PI,0]			# Trot
		self.period = 2/2
		#self.SteNum = 32							#32 # Devide 2*PI to multiple steps
		self.fre_cyc = fre#1.25#0.80
		self.SteNum = int(1/(timeStep*self.fre_cyc))
		self.spinePhase = self.phaseDiff[2]
		# --------------------------------------------------------------------- #
		self.spine_A = 30#30
		#print("angle --> ", self.spine_A)
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

	def getLegCtrl(self, leg_M, curStep, leg_ID):
		curStep = curStep % self.SteNum
		turnAngle = self.turn_F
		leg_flag = "F"
		if leg_ID > 1:
			leg_flag = "H"
			turnAngle = self.turn_H
		radian = 2*np.pi * curStep/self.SteNum

		currentPos = self.pathStore.getOvalPathPoint(radian, leg_flag, self.period)
		trg_x = currentPos[0]
		trg_y = currentPos[1]
		tX = math.cos(turnAngle)*trg_x - math.sin(turnAngle)*trg_y
		tY = math.cos(turnAngle)*trg_y + math.sin(turnAngle)*trg_x
		qVal = leg_M.pos_2_angle(tX, tY)
		#qVal = leg_M.pos_2_angle(trg_x, trg_y)
		return qVal

	def getSpineVal(self, spineStep):
		spinePhase = 2*np.pi*spineStep/self.SteNum
		return self.spine_A*math.sin(spinePhase)#+self.spine_A

	def runStep(self):
		foreLeg_left_q = self.getLegCtrl(self.fl_left, 
			self.curStep + self.stepDiff[0], 0)
		foreLeg_right_q = self.getLegCtrl(self.fl_right, 
			self.curStep + self.stepDiff[1], 1)
		hindLeg_left_q = self.getLegCtrl(self.hl_left, 
			self.curStep + self.stepDiff[2], 2)
		hindLeg_right_q = self.getLegCtrl(self.hl_right, 
			self.curStep + self.stepDiff[3], 3)

		spineStep = self.curStep + self.stepDiff[4]
		spine = -self.getSpineVal(spineStep)
		#spine = 0
		self.curStep = (self.curStep + 1) % self.SteNum

		#foreLeg_left_q = [0,0]
		#foreLeg_right_q = [0,0]
		#hindLeg_left_q = [0,0]
		#hindLeg_right_q = [0,0]

		leg_ctrl = []
		leg_ctrl.extend(foreLeg_left_q)
		leg_ctrl.extend(foreLeg_right_q)
		leg_ctrl.extend(hindLeg_left_q)
		leg_ctrl.extend(hindLeg_right_q)
		spine_ctrl = spine
		head_ctrl = 0
		tail_ctrl = 0#math.pi/6
		
		return leg_ctrl, spine_ctrl, head_ctrl, tail_ctrl



		