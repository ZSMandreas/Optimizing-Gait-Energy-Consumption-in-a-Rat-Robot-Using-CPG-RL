import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import math

def getPathData(flag):
	filePath = "Data/path_"+flag+".txt"
	inputFile = open(filePath,"r")
	dataList = []

	inputLines = inputFile.readlines()
	for line in inputLines:
		lineStrData = line.split(' ')
		dataNum = len(lineStrData) - 1
		lineData = []
		for i in range(dataNum):
			lineData.append(float(lineStrData[i]))
		dataList.append(lineData)

	# --------------------- #
	tL = len(dataList)
	returnList = [[],[],[]]
	for i in range(tL):
		returnList[0].append(dataList[i][0])
		returnList[1].append(dataList[i][1])
		returnList[2].append(dataList[i][2])
	# --------------------- #	
	return returnList

if __name__ == '__main__':
	for_alex = getPathData("alex_080")
	for_own = getPathData("own_080")

	plt.plot(for_alex[1], for_alex[2], label='From Alex')
	plt.plot(for_own[1], for_own[2], label='From Yuhong')
	plt.legend(loc="upper left", fontsize=15)
	plt.title('Vertical change of robot center', fontsize=15)
	plt.xlabel('y-coordinate (m)', fontsize=15)
	plt.ylabel('z-coordinate (m)', fontsize=15)
	plt.savefig('Longitudinal_080.png')
	#plt.axis('equal')
	plt.grid()
	plt.show()


	plt.plot(for_alex[1], for_alex[0], color='orange', label='From Alex')
	plt.plot([for_alex[1][0],for_alex[1][-1]], [for_alex[0][0],for_alex[0][-1]], \
		 color='orange',linestyle='--')
	plt.plot(for_own[1], for_own[0], color='blue', label='From Yuhong')
	plt.plot([for_own[1][0],for_own[1][-1]], [for_own[0][0],for_own[0][-1]], \
		 color='blue',linestyle='--')
	plt.legend(loc="upper left", fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title('Horizontal change of robot center', fontsize=15)
	plt.xlabel('y-coordinate (m)', fontsize=15)
	plt.ylabel('x-coordinate (m)', fontsize=15)
	plt.axis('equal')
	plt.grid()
	plt.subplots_adjust(left=0.15, bottom=0.15,right=0.95, top=0.93,hspace=0.1,wspace=0.1)
	plt.savefig('Lateral_080.png')
	plt.show()

