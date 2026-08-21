from Kinematics import *
from ToDefine import *
import math

class CMouseLeg(CKinematics):
	"""docstring for CMouseLeg"""
	_typPhase = ["Swing", "Stance"]

	_ptLeg = CKoord()	# x/y destination pos in move, current pos !in move
	_dgNext = CLegPos()	# Next Point on walking line
	_output = CLegPos()	# Output Value
	_currPhase = ""
	_docu = []			#docu array
	# VARIABLES
	# vector from current to destination x/y-point, divided by step stepcount
	_riseStep = 0
	_risetime = 0	# time till the leg is lifted/set down
	# ------ #
	_stepcount = 0	# number of kinematic-steps
	_step = 0		# current step number

	def __init__(self, leg, side, pawLift):
		super(CMouseLeg, self).__init__()
		self.vx = 0
		self.vy = 0
		
		self.leg = leg
		self.side = side
		self.pawLift = pawLift		# how high does the leg lift

		for i in range(100):
			self._docu.append(CKoord())

	# ------------------------------------------------------------------------ #
	def StepStart(self, x, y):
		self._step = 0											# cast: Abschneiden gewollt s. jobTime
		if self._stepcount <= 1:
			self.MoveTo(x, y)									# if Dist < Resolution then Singlestep.
		else:
			self.vx = (x - self._ptLeg.x) / self._stepcount		# compute step vector from known position (= at start!)
			self.vy = (y - self._ptLeg.y) / self._stepcount
			self._dgNext = self.NextWayPoint()								# first step of kinematik move

	def StepNext(self):
		self._step = self._step + 1
		if self._step > self._stepcount: 
			return False
		self._output = self.SetPosition(self._dgNext)	# vorausberechneten Punkt ausgeben
		if self._step < self._stepcount:				# wenn nicht letzter Punkt:
			self._dgNext = self.NextWayPoint()			# neuen Punkt berechnen, solange Motor l�uft
		return True										# weiter gehts

	def NextWayPoint(self):
		X = self._ptLeg.x + self.vx
		Y = self._ptLeg.y + self.vy
		if (self._currPhase == self._typPhase[0]) and (self._step > 0) \
			and (self._step < (self._stepcount - 1)) and self.leg == 'h':
			if self._step < self._risetime:								# leg rises
				Y = Y + self._riseStep * self._step						# rise stepwise with time
			elif self._step > (self._stepcount - self._risetime):		# leg is set down
				Y = Y + self._riseStep * (self._stepcount - self._step)	# set down with time
			else:
				Y = Y + self.pawLift									# else keep height

		# update initial positions (closed loop control)
		self.lhlCoilInit = initCoil[0].lhl
		self.lflCoilInit = initCoil[0].lfl
		self.rhlCoilInit = initCoil[0].rhl
		self.rflCoilInit = initCoil[0].rfl
		#print(self.lhlCoilInit, " ", self.lflCoilInit, self. rhlCoilInit, self.rflCoilInit)

		if self.leg == 'f':
			return self.ikforeleg(X, Y, self.side)
		else:
			return self.ikhindleg(X, Y, self.side);

	def SetPosition(self, ang): #double deg1, double deg2);
		self._docu[self._step] = self._ptLeg; 	# documentation for debugging
		if (self._currPhase == self._typPhase[0]) and (self._step > 0) \
			and (self._step < (self._stepcount - 1)) and self.leg == 'f':
			if self.side == 'l':
				ang.coil = ang.coil - (self.pawLift * 2)
			else:
				ang.coil = ang.coil + (self.pawLift * 2)
		self._ptLeg.x = self._ptLeg.x + self.vx;
		self._ptLeg.y = self._ptLeg.y + self.vy;		# Vektor auf letzten Punkt addieren
		return ang										# output zurückgeben

	def Distance(self, x1, y1, x2, y2):
		return math.hypot((x2 - x1), (y2 - y1));

	# ------------------------------------------------------------------------ #

	def StartLeg(self, x, y, length, thePhase):
		self._stepcount = length
		self._currPhase = thePhase
		if (thePhase == self._typPhase[0]):
			self._risetime = int(round(float(length) / 5))
			self._riseStep = self.pawLift / self._risetime
		self.StepStart(x, y)

	def MoveTo(self, x, y):
		self.vx = 0
		self.vy = 0
		self._step = 0
		self._stepcount = 0
		self._ptLeg = CKoord(x, y)
		# Next waypoint berechnet Kinematik und in Output Speichern.
		self._output = self.SetPosition(self.NextWayPoint())
		
	def GetNext(self):
		if self.StepNext():
			return self._output
		else:	# currently always same output. when done, then always endposition
			return self._output

	def Dump(self):
		print(self.leg, "; ", self.side, "; ", self.pawLift)
		for i in range(self._stepcount):
			print(self._docu[i].x, "; ", self._docu[i].y)