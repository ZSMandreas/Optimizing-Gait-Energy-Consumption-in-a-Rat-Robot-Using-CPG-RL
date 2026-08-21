import serial

class Port(object):
	"""docstring for Port"""
	def __init__(self, SerialPort, BaudRate):
		super(Port, self).__init__()
		self.theUART = serial.Serial(SerialPort, BaudRate, timeout=0.5)

	def sendUartMessage(self, message):
		self.theUART.write(message.encode())

	def shutdown(self):
		self.theUART.close()


