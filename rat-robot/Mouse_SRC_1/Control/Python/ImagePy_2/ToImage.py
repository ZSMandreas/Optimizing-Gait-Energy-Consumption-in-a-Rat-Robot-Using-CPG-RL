import socket
import cv2
import numpy

class CImage(object):
	"""docstring for CImage"""
	def __init__(self, theSocket):
		super(CImage, self).__init__()
		self.theSocket = theSocket
		self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),15]

		self.blackLower = (0, 0, 0)
		self.blackUpper = (180, 255, 46)
		
		self.redLower = (156, 43, 46)
		self.redUpper = (180, 255, 255)

		self.colorLower = self.redLower
		self.colorUpper = self.redUpper

	def sendImage(self, frame_1, frame_2):
		result_1, imgencode_1 = cv2.imencode('.jpg', frame_1, self.encode_param)
		result_2, imgencode_2 = cv2.imencode('.jpg', frame_2, self.encode_param)

		data_1 = numpy.array(imgencode_1)
		data_2 = numpy.array(imgencode_2)
		stringData_1 = data_1.tostring()
		stringData_2 = data_2.tostring()

		self.theSocket.send(str.encode(str(len(stringData_1)).ljust(16)))
		self.theSocket.send(stringData_1);
		self.theSocket.send(str.encode(str(len(stringData_2)).ljust(16)))
		self.theSocket.send(stringData_2);

		receive = self.theSocket.recv(1024)

	def getPos(self, theFrame):
		blurred = cv2.GaussianBlur(theFrame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.colorLower, self.colorUpper)

		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		cnts,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#print(cnts)
		center = None

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use it to compute the minimum enclosing circle
			# and centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame, then update the list of tracked points
				cv2.circle(theFrame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
				cv2.circle(theFrame, center, 5, (0, 0, 255), -1)

		return theFrame, center
