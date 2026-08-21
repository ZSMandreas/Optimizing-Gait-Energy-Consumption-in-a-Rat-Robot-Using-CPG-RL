from MotorCtrl import *
from ToImage import *
from threading import Thread
import time

import socket
import cv2
import numpy


def videoInfo(tInfo, tSock):
	capture_1 = cv2.VideoCapture(0)
	#capture_2 = cv2.VideoCapture(2)
	ret_1, frame_1 = capture_1.read()
	#ret_2, frame_2 = capture_2.read()
	opImage = CImage(tSock[0])
	frame_2 = "test"
	while ret_1:
		tCmd = tInfo[0]
		if tCmd == "q":
			break
		time.sleep(0.5)
		if tCmd == "t":
			frame_1, center = opImage.getPos(frame_1)
			iSize = frame_1.shape
			if center != None:
				origin = iSize[1] / 2
				#print(center, iSize)
				curPos = (center[0] - origin) / origin 
				tInfo[1] = curPos

		opImage.sendImage(frame_1, frame_2)
		ret_1, frame_1 = capture_1.read()
		#ret_2, frame_2 = capture_2.read()
		if cv2.waitKey(10) == 27:
			break

def cmdSend(tInfo):
	SerialPort = "/dev/ttyAMA0"
	BaudRate = 1000000

	toMotors = Motors(SerialPort, BaudRate)
	oldCmd = "i"
	while True:
		time.sleep(0.05)
		tCmd = tInfo[0]
		targetPos = tInfo[1]
		endFlag = toMotors.ctrl(tCmd, targetPos)
		if endFlag:
			break

def cmdInput(tInput):
	while True:
		user_input = input('Type user input: ')
		tInput[0] = user_input
		if user_input == "q":
			break

if __name__ == '__main__':
	address = ('192.168.137.11', 8080)
	try:
		sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		sock.connect(address)
	except socket.error as msg:
		print("The Server is not found !!!")
		print(msg)
		sys.exit(1)

	tlist = ["i", 0]
	tSock = [sock]
	t1 = Thread(target=cmdSend, args=(tlist,))
	t2 = Thread(target=cmdInput, args = (tlist,))
	t3 = Thread(target=videoInfo, args = (tlist, tSock))

	t1.start()
	t2.start()
	t3.start()

	t2.join()  # interpreter will wait until your process get completed or terminated

	print('The end')