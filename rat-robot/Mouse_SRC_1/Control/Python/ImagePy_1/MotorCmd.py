from TheSerial import *
import time 

MAX_RECIEVE_LENGTH = 255
MAX_ARG_LENGTH = 50
MAX_ARG_PER_LINE = 10

class Ccmnd(object):
	"""docstring for Ccmnd"""
	def __init__(self):
		self.valid = False
		self.val1 = 0
		self.val2 = 0
		self.val3 = 0
		self.command = 7	# FIXIT for TypCmd

class CMouseCom(object):
	"""docstring for CMouseCom"""
	Motors = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 24]
	MotorP = 17
	MotorI = 0
	MotorD = 35

	left_is_free = True
	right_is_free = True;
	storageBuffer = 10000;
	storageSensorBoards = 4			# 01 -> 0 / 03 -> 1 / 11 -> 2 / 13 -> 3
	storageVariablesFoot = 1		# value
	storageVariablesKnee = 2		# BX, BY

	storageServoBoards = 13
	storageVariablesPos = 2
	storageVariablesPID = 3

	def __init__(self, SerialPort, BaudRate):
		super(CMouseCom, self).__init__()
		self.toUART = Port(SerialPort, BaudRate)

		self.StoreArraySensor = [[[0 for i in range(self.storageBuffer)]\
			for j in range (self.storageVariablesKnee + self.storageVariablesFoot)]\
			for k in range(self.storageSensorBoards)]
		self.sensor_index = [[0,0] for i in range(self.storageSensorBoards)]

		self.StoreArrayPos = [[[0 for i in range(self.storageBuffer)]\
			for j in range(self.storageVariablesPos)] for k in range(self.storageServoBoards)]
		self.StoreArrayPID = [[[0 for i in range(self.storageBuffer)]\
			for j in range(self.storageVariablesPID)] for k in range(self.storageServoBoards)]
		self.position_index = [0]*self.storageServoBoards
		self.pid_index =[0]*self.storageServoBoards
	def shutdown(self):
		self.toUART.shutdown()

	def clearSensorComplete(self):
		for i in range(self.storageSensorBoards):
			for j in range(3):
				for k in range(self.storageBuffer):
					self.StoreArraySensor[i][j][k] = 0
			self.sensor_index[i][0] = 0;
			self.sensor_index[i][1] = 0;


	def setConsoleCcmnd(self, cmd, val1 = 0, val2 = 0, val3 = 0):
		self.ReceiveMsg(cmd.command, val1, val2, val3);
	
	def ProcessSpine(self, cmd, val1, val2 = 0, val3 = 0):
		if cmd == "SetMotorPos":
			self.sendMotorSeial(val1, val2, val3)
		elif cmd == "GetSensorValue":
			self.sendSensorRequest(val1, val2)
		elif cmd == "SetLed":
			self.setMotorLed(val1, val2)
		elif cmd == "SetMotorOff":
			self.setMotorOFF(val1)
		elif cmd == "MPwrOff":
			self.setMotorPwrOFF()
		else:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print("Error - Unknwon UART Send Request!")

	#OLD COMMUNICATION
	def setMotorOFF(self, tId):
		msg = ":%02d!P=OFF\n"%(tId)
		self.toUART.sendUartMessage(msg)
	def setMotorPwrOFF(self):
		msg = "!PS-\n"
		self.toUART.sendUartMessage(msg)
	def sendNL(self):
		msg = "\n"
		self.toUART.sendUartMessage(msg)
	def MotorPwrCycle(self):
		msg = "!PS-\n"
		self.toUART.sendUartMessage(msg)
		time.sleep(0.5)
		msg = "!PS-\n"
		self.toUART.sendUartMessage(msg)

	def setMotorPID(self):
		for i in range(self.storageServoBoards):
			msg = ":%02d!CP=%03x0\n"%(self.Motors[i], self.MotorP)
			self.toUART.sendUartMessage(msg)
			time.sleep(0.005)
			msg = ":%02d!CI=%03x0\n"%(self.Motors[i], self.MotorI)
			self.toUART.sendUartMessage(msg)
			time.sleep(0.005)
			msg = ":%02d!CD=%03x0\n"%(self.Motors[i], self.MotorD)
			self.toUART.sendUartMessage(msg)
			time.sleep(0.005)

	def setMotorSilent(sefl, tId, val1):
		msg = "\n"
		if val1 == 1:		# switch on
			msg = ":%02d!U+\n"%(tId)
		elif val1 == 0:		# switch off
			msg = ":%02d!U-\n"%(tId)
		self.toUART.sendUartMessage(msg)
			
	def MotorSetup(self):
		print("Nothing to do !!!")

	def ReceiveMsg(self, cmd, val1 = 0, val2 = 0, val3 = 0, val4 = 0):
		index = self.IDtoStoreArrayIndex(val1)
		if cmd == "KneeSensorValue":
			if self.sensor_index[index][0] >= self.storageBuffer:
				self.clearKneePart(index)
			if val2 >= 2048:
				val2 = val2-4096
			if val3 >= 2048:
				val3 = val2-4096
			self.StoreArraySensor[index][0][self.sensor_index[index][0]] = val2
			self.StoreArraySensor[index][1][self.sensor_index[index][0]] = val3
			self.sensor_index[index][0] = self.sensor_index[index][0] + 1
		elif cmd == "FootSensorValue":
			if self.sensor_index[index][1] >= self.storageBuffer:
				self.clearFootPart(index)
			self.StoreArraySensor[index][2][self.sensor_index[index][1]] = val2
			self.sensor_index[index][1] += 1

	# Semaphore for sensor value requests on SWU
	def setLeftBlock(self):
		self.left_is_free = False
	def setRightBlock(self):
		self.right_is_free = False
	# ------------------------------------------------------------ #
	# ------------------------------------------------------------ #
	def MouseInputLoop(self):
		self.receiveData()

	def sendMotorSeial(self, tID, pos, speed):
		msg = ":%02d!P=%04x\n"%(tID, pos)
		self.toUART.sendUartMessage(msg)
	def setMotorLed(self, tID, state): 	# 0,1,2
		msg = "\n"
		if state == 1:		# Switch on
			msg = ":%02d!L+\n"%(tID)
		elif state == 0:
			msg = ":%02d!L-\n"%(tID)
		else:
			msg = ":%02d!L=%04x\n"%(tID, state)
		self.toUART.sendUartMessage(msg)

	def sendStreamRequest(self, tID, frequency, amount):
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("Error - sendStreamRequest not implemented!")

	def sendSensorRequest(self, tID, sensor):
		msg = "\n"
		if sensor == 0:
			msg = ":%02d?K\n"%(tID)
		elif sensor == 1:
			msg = ":%02d?F\n"%(tID)
		elif sensor == 2:
			msg = ":%02d?P\n"%(tID)
		elif sensor == 3:
			msg = ":%02d?C\n"%(tID)
		self.toUART.sendUartMessage(msg)

	def receiveData(self):			# give array with MAX_ARG_LENGTH
		pass
	def checkComndConsole(self):
		pass

	def IDtoStoreArrayIndex(self, tID):
		if tID == 1:
			return 0
		if tID == 3:
			return 1
		if tID == 11:
			return 2
		if tID == 13:
			return 3
	def clearFootPart(self, index):
		for i in range(self.storageBuffer):
			self.StoreArraySensor[index][2][i] = 0
		self.sensor_index[index][1] = 0

	def clearKneePart(self, index):
		for i in range(self.storageBuffer):
			self.StoreArraySensor[index][0][i] = 0;
			self.StoreArraySensor[index][1][i] = 0;
		self.sensor_index[index][0] = 0

	def convertHexToInt(self, digit):
		tHex = '0x'+str(digit)
		return int(tHex,16) 

	def convertToInt(self, val, length):
		return self.convertHexToInt(val)