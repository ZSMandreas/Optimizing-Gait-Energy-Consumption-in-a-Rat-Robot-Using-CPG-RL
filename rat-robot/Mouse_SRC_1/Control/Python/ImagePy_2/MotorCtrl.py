from ToDefine import *

from ToLeg import *
from ToSpine import *
from MotorCmd import *

import time
import math

class Motors(CMouseCom):
	"""docstring for Motors"""
	MotorNum = 13
	CommandDelay = 0.015#0.015
	FLift = 35
	HLift = 30


	uFrontLegStart = -10		# x start pos. of pace
	uHindLegStart = -20
	uStepLengthF = 60			# length of one leg move on the ground
	uStepLengthH = 60			# length of one leg move on the ground
	uHWalkLevel = 0				# y walking level of init and walking
	uFWalkLevel = 10			# y walking level of init and walking
	uSitting_x = -30			# X position for foot in sitting
	uSitting_y = 30				# Y position for foot in sitting
	uPosSpineFlexRelax = 180	# - is flex  NO+!!
	uPosHeadPan = 180
	uPosHeadTilt = 202
	TrottArray = [0]*(MotorNum+1)
	def __init__(self, SerialPort, BaudRate):
		super(Motors, self).__init__(SerialPort, BaudRate)
		self.LForeLeft = CMouseLeg('f', 'l', self.FLift)
		self.LForeRight = CMouseLeg('f', 'r', self.FLift)
		self.LHindLeft = CMouseLeg('h', 'l', self.HLift)
		self.LHindRight = CMouseLeg('h', 'r', self.HLift)
		self.Spine = CSpine()

		self.labelFlag = 0
		self.motionlength = 50
		self.margin = 1

	def ctrl(self, cmd, targetPos):
		if cmd == "q":
			self.shutdown()
			return True
		if self.labelFlag == 0:
			if cmd == "i":
				tList = self.toInit()
				self.publish(tList)
			elif cmd == 't':
				tList = self.toShakingHead(targetPos)
				self.publish(tList)
			elif cmd == '+':
				self.CommandDelay = self.CommandDelay-0.001
				if self.CommandDelay <= 0.005:
					self.CommandDelay  = 0.005
			elif cmd == '-':
				self.CommandDelay = self.CommandDelay+0.001
				if self.CommandDelay >= 0.04:
					self.CommandDelay  = 0.04
			elif cmd == 'w':
				self.labelFlag = 1
				tList = self.toMove("w")
				self.publish(tList)

		if self.labelFlag == 1:
			if cmd == "e":
				self.labelFlag = 0
				tList = self.toInit()
				self.publish(tList)
			elif cmd == 'w':
				tList = self.toMove("w")
				self.publish(tList)
			elif cmd == 's':
				tList = self.toMove("s")
				self.publish(tList)
			elif cmd == 'a':
				tList = self.toMove("a")
				self.publish(tList)
			elif cmd == 'd':
				tList = self.toMove("d")
				self.publish(tList)

		return False


	def publish(self, tList):
		tL = len(tList)
		for i in range(tL):
			self.SendMotorMsgs(tList[i])
			time.sleep(self.CommandDelay)

	def toInit(self):
		self.clearArr()
		initCoil[0].lhl = 180
		initCoil[0].lfl = 180
		initCoil[0].rhl = 180
		initCoil[0].rfl = 180
		self.LHindLeft.StartLeg(0, 0, 1, "Stance")
		self.LHindRight.StartLeg(0, 0, 1, "Stance")
		self.LForeLeft.StartLeg(0, 0, 1, "Stance")
		self.LForeRight.StartLeg(0, 0, 1, "Stance")

		tmpSpine = self.Spine.centre()

		self.TrottArray[A_TIMESTAMP] = 0
		tmpLeg = self.LHindLeft.GetNext()
		self.TrottArray[A_HINDLEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDLEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LHindRight.GetNext();
		self.TrottArray[A_HINDRIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDRIGHT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeLeft.GetNext();
		self.TrottArray[A_FORELEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORELEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeRight.GetNext();
		self.TrottArray[A_FORERIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORERIGHT_KNEE] = tmpLeg.coil
		self.TrottArray[A_SPINE] = tmpSpine.spine
		self.TrottArray[A_TAIL] = tmpSpine.tail
		self.TrottArray[A_SPINE_FLEX] = 180
		self.TrottArray[A_HEAD_PAN] = self.uPosHeadPan
		self.TrottArray[A_HEAD_TILT] = self.uPosHeadTilt

		tempList =  list(self.TrottArray)
		return [tempList]

	def toShakingHead(self, curPos):
		maxAngle = 30
		theLevel = round((curPos + 1) / 0.4)
		turnAngle = (theLevel*0.4 - 1)*maxAngle


		initCoil[0].lhl = 180
		initCoil[0].lfl = 180
		initCoil[0].rhl = 180
		initCoil[0].rfl = 180
		self.LHindLeft.StartLeg(0, 0, 1, "Stance")
		self.LHindRight.StartLeg(0, 0, 1, "Stance")
		self.LForeLeft.StartLeg(0, 0, 1, "Stance")
		self.LForeRight.StartLeg(0, 0, 1, "Stance")

		tmpSpine = self.Spine.centre()

		self.TrottArray[A_TIMESTAMP] = 0
		tmpLeg = self.LHindLeft.GetNext()
		self.TrottArray[A_HINDLEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDLEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LHindRight.GetNext();
		self.TrottArray[A_HINDRIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDRIGHT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeLeft.GetNext();
		self.TrottArray[A_FORELEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORELEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeRight.GetNext();
		self.TrottArray[A_FORERIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORERIGHT_KNEE] = tmpLeg.coil
		self.TrottArray[A_SPINE] = tmpSpine.spine
		self.TrottArray[A_TAIL] = tmpSpine.tail
		self.TrottArray[A_SPINE_FLEX] = 180
		self.TrottArray[A_HEAD_PAN] = self.uPosHeadPan + turnAngle
		self.TrottArray[A_HEAD_TILT] = self.uPosHeadTilt

		tempList =  list(self.TrottArray)
		return [tempList]
	
	def toMove(self, moveFlag):
		if moveFlag == "s":
			return []
		halfMotion = int(round(self.motionlength / 2))
		tail = False #True
		tmpSpine = self.Spine.centre()
		
		'''
		leftFlag = 1.2
		rightFlag = 0.8
		if moveFlag == "w":
			tmpSpine = self.Spine.centre()
		elif moveFlag == "a":
			tmpSpine = self.Spine.moveStepLeft(self.motionlength)
		elif moveFlag == "d":
			tmpSpine = self.Spine.moveStepRight(self.motionlength)
		else:
			tmpSpine = self.Spine.centre()
		'''
		#"""
		leftFlag = 1
		rightFlag = 1
		if moveFlag == "w":
			leftFlag = 0.9
			rightFlag = 1.1
		if moveFlag == "a":
			leftFlag = 1.6
			rightFlag = 0.4
		if moveFlag == "d":
			leftFlag = 0.4
			rightFlag = 1.6
		#"""

		self.LHindLeft.StartLeg(self.uHindLegStart*self.margin, self.uHWalkLevel,\
			halfMotion*leftFlag, "Swing")
		#self.LHindRight.StartLeg(0, 0, 1, "Stance")
		#self.LForeLeft.StartLeg(0, 0, 1, "Stance")
		#self.LForeRight.StartLeg(0, 0, 1, "Stance")
		#"""
		self.LHindRight.StartLeg((self.uStepLengthH + self.uHindLegStart)*self.margin, self.uHWalkLevel,\
			halfMotion*rightFlag, "Stance")
		self.LForeLeft.StartLeg((self.uStepLengthF + self.uFrontLegStart)*self.margin, self.uFWalkLevel, \
			halfMotion*leftFlag, "Stance")
		self.LForeRight.StartLeg(self.uFrontLegStart*self.margin, self.uFWalkLevel, halfMotion*rightFlag, "Swing")
		#"""
		ctrlList = []
		for i in range(halfMotion):
			self.TrottArray[A_TIMESTAMP] = i
			tmpLeg = self.LHindLeft.GetNext()
			self.TrottArray[A_HINDLEFT_HIP] = tmpLeg.leg
			self.TrottArray[A_HINDLEFT_KNEE] = tmpLeg.coil
			tmpLeg = self.LHindRight.GetNext()
			self.TrottArray[A_HINDRIGHT_HIP] = tmpLeg.leg
			self.TrottArray[A_HINDRIGHT_KNEE] = tmpLeg.coil
			tmpLeg = self.LForeLeft.GetNext()
			self.TrottArray[A_FORELEFT_HIP] = tmpLeg.leg
			self.TrottArray[A_FORELEFT_KNEE] = tmpLeg.coil
			tmpLeg = self.LForeRight.GetNext()
			self.TrottArray[A_FORERIGHT_HIP] = tmpLeg.leg
			self.TrottArray[A_FORERIGHT_KNEE] = tmpLeg.coil
			self.TrottArray[A_SPINE] = tmpSpine.spine
			if tail:
				self.TrottArray[A_TAIL] = self.Spine.moveTailLeft(halfMotion)
			else:
				self.TrottArray[A_TAIL] = tmpSpine.tail
			self.TrottArray[A_SPINE_FLEX] = self.Spine.stretch();
			self.TrottArray[A_HEAD_PAN] = self.uPosHeadPan
			self.TrottArray[A_HEAD_TILT] = self.uPosHeadTilt

			tempList =  list(self.TrottArray)
			ctrlList.append(tempList)

		self.LHindLeft.StartLeg((self.uStepLengthH + self.uHindLegStart)*self.margin, self.uHWalkLevel,\
			halfMotion*leftFlag, "Stance");
		#self.LHindRight.StartLeg(0, 0, 1, "Stance")
		#self.LForeLeft.StartLeg(0, 0, 1, "Stance")
		#self.LForeRight.StartLeg(0, 0, 1, "Stance")
		#"""
		self.LHindRight.StartLeg(self.uHindLegStart*self.margin, self.uHWalkLevel,\
			halfMotion*rightFlag, "Swing");
		self.LForeLeft.StartLeg(self.uFrontLegStart*self.margin, self.uFWalkLevel, halfMotion*leftFlag, "Swing");
		self.LForeRight.StartLeg((self.uStepLengthF + self.uFrontLegStart)*self.margin, self.uFWalkLevel, \
			halfMotion*rightFlag, "Stance");
		#"""
		for i in range(halfMotion):
			self.TrottArray[A_TIMESTAMP] = i
			tmpLeg = self.LHindLeft.GetNext()
			self.TrottArray[A_HINDLEFT_HIP] = tmpLeg.leg
			self.TrottArray[A_HINDLEFT_KNEE] = tmpLeg.coil
			tmpLeg = self.LHindRight.GetNext()
			self.TrottArray[A_HINDRIGHT_HIP] = tmpLeg.leg
			self.TrottArray[A_HINDRIGHT_KNEE] = tmpLeg.coil
			tmpLeg = self.LForeLeft.GetNext()
			self.TrottArray[A_FORELEFT_HIP] = tmpLeg.leg
			self.TrottArray[A_FORELEFT_KNEE] = tmpLeg.coil
			tmpLeg = self.LForeRight.GetNext()
			self.TrottArray[A_FORERIGHT_HIP] = tmpLeg.leg
			self.TrottArray[A_FORERIGHT_KNEE] = tmpLeg.coil
			self.TrottArray[A_SPINE] = tmpSpine.spine
			if tail:
				self.TrottArray[A_TAIL] = self.Spine.moveTailRight(halfMotion)
			else:
				self.TrottArray[A_TAIL] = tmpSpine.tail
			self.TrottArray[A_SPINE_FLEX] = self.Spine.stretch();
			self.TrottArray[A_HEAD_PAN] = self.uPosHeadPan
			self.TrottArray[A_HEAD_TILT] = self.uPosHeadTilt

			tempList =  list(self.TrottArray)
			ctrlList.append(tempList)
		return ctrlList

	def clearArr(self):
		centrePos = 180
		#initalize Leg motion with Right leg forward
		self.LHindLeft.StartLeg(0, 0, 1, "Stance")
		self.LHindRight.StartLeg(0, 0, 1, "Stance")
		self.LForeLeft.StartLeg(0, 0, 1, "Stance")
		self.LForeRight.StartLeg(0, 0, 1, "Stance")

		self.TrottArray[A_TIMESTAMP] = 0
		tmpLeg = self.LHindLeft.GetNext()
		self.TrottArray[A_HINDLEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDLEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LHindRight.GetNext()
		self.TrottArray[A_HINDRIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_HINDRIGHT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeLeft.GetNext()
		self.TrottArray[A_FORELEFT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORELEFT_KNEE] = tmpLeg.coil
		tmpLeg = self.LForeRight.GetNext()
		self.TrottArray[A_FORERIGHT_HIP] = tmpLeg.leg
		self.TrottArray[A_FORERIGHT_KNEE] = tmpLeg.coil
		self.TrottArray[A_SPINE] = centrePos
		self.TrottArray[A_TAIL] = centrePos
		self.TrottArray[A_SPINE_FLEX] = centrePos
		self.TrottArray[A_HEAD_PAN] = self.uPosHeadPan
		self.TrottArray[A_HEAD_TILT] = self.uPosHeadTilt

	def toRemap(self, val):
		maxPos = 4095
		maxDeg = 360
		tMap = (maxPos * val) / maxDeg
		return int(abs(tMap))

	def SendMotorMsgs(self, tList):
		SetMotorPos = "SetMotorPos"
		self.ProcessSpine(SetMotorPos, ID_FORELEFT_HIP, self.toRemap(tList[A_FORELEFT_HIP]), 1)
		self.ProcessSpine(SetMotorPos, ID_FORELEFT_KNEE, self.toRemap(tList[A_FORELEFT_KNEE]), 1)
		self.ProcessSpine(SetMotorPos, ID_FORERIGHT_HIP, self.toRemap(tList[A_FORERIGHT_HIP]), 1)
		self.ProcessSpine(SetMotorPos, ID_FORERIGHT_KNEE, self.toRemap(tList[A_FORERIGHT_KNEE]), 1)
		self.ProcessSpine(SetMotorPos, ID_HINDLEFT_HIP, self.toRemap(tList[A_HINDLEFT_HIP]), 1)
		self.ProcessSpine(SetMotorPos, ID_HINDLEFT_KNEE, self.toRemap(tList[A_HINDLEFT_KNEE]), 1)
		self.ProcessSpine(SetMotorPos, ID_HINDRIGHT_HIP, self.toRemap(tList[A_HINDRIGHT_HIP]), 1)
		self.ProcessSpine(SetMotorPos, ID_HINDRIGHT_KNEE, self.toRemap(tList[A_HINDRIGHT_KNEE]), 1)
		self.ProcessSpine(SetMotorPos, ID_SPINE, self.toRemap(tList[A_SPINE]), 1)
		self.ProcessSpine(SetMotorPos, ID_TAIL, self.toRemap(tList[A_TAIL]), 1)
		self.ProcessSpine(SetMotorPos, ID_SPINE_FLEX, self.toRemap(tList[A_SPINE_FLEX]), 1)
		self.ProcessSpine(SetMotorPos, ID_HEAD_PAN, self.toRemap(tList[A_HEAD_PAN]), 1)
		self.ProcessSpine(SetMotorPos, ID_HEAD_TILT, self.toRemap(tList[A_HEAD_TILT]), 1)