import socket
import time
import cv2
import numpy

def ReceiveVideo():
	address = ('0.0.0.0', 8080)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(address)
	s.listen(1)

	def recvall(sock, count):
		buf = b''
		while count:
			newbuf = sock.recv(count)
			if not newbuf: return None
			buf += newbuf
			count -= len(newbuf)
		return buf
	conn, addr = s.accept()
	print('connect from:'+str(addr))

	SERVER_1 = "SERVER_1"
	cv2.namedWindow(SERVER_1, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(SERVER_1, 640, 480)

	SERVER_1 = "SERVER_2"
	cv2.namedWindow(SERVER_2, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(SERVER_2, 640, 480)

	while 1:
		start = time.time()
		length_1 = recvall(conn,16)
		stringData_1 = recvall(conn, int(length_1))
		data_1 = numpy.frombuffer(stringData_1, numpy.uint8)
		decimg_1=cv2.imdecode(data_1,cv2.IMREAD_COLOR)

		length_2 = recvall(conn,16)
		stringData_2 = recvall(conn, int(length_2))
		data_2 = numpy.frombuffer(stringData_2, numpy.uint8)
		decimg_2=cv2.imdecode(data_2,cv2.IMREAD_COLOR)

		cv2.imshow(SERVER_1, decimg_1)
		cv2.imshow(SERVER_2,decimg_2)

		end = time.time()
		seconds = end - start
		fps = 1/seconds
		conn.send(bytes(str(int(fps)),encoding='utf-8'))
		k = cv2.waitKey(10)&0xff

	s.close()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	ReceiveVideo()