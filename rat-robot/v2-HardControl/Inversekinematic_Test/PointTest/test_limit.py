import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import math
from legs import LegModel

def readPoints(file_name):
	inputFile = open(file_name,"r")
	dataList = []
	inputLines = inputFile.readlines()
	for line in inputLines:
		lineStrData = line.split(' ')
		dataNum = len(lineStrData) - 1
		lineData = []
		for i in range(dataNum):
			lineData.append(float(lineStrData[i]))
		dataList.append(lineData)

	return dataList, len(inputLines)

if __name__ == '__main__':
	trg_range, points_num = readPoints('/home/user/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/Inversekinematic_Test/PointTest/feasible_point.txt')
	py_list = [trg_range[j][0] for j in range(points_num)]
	pz_list = [trg_range[j][1] for j in range(points_num)]

	trg_knee, k_points_num = readPoints('/home/user/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/Inversekinematic_Test/PointTest/knee_point.txt')
	k_py_list = [trg_knee[j][0] for j in range(k_points_num)]
	k_pz_list = [trg_knee[j][1] for j in range(k_points_num)]

	#plt.show()

	# 1: AE 	2: AB
	# 3: BC 	4: CD
	# 5: DE 	6: EF (DE+EF ~= DF)
	leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
	#leg_params=[0.031, 0.0088, 0.0128, 0.040, 0.015, 0.035]
	leg_M = LegModel(leg_params)

	endPoints = []
	for i in range(points_num):
		qVal = leg_M.pos_2_angle(py_list[i], pz_list[i])
		#print(qVal)
		if len(qVal) == 0:
			continue
		cur_points = leg_M.angel_2_pos(qVal[0], qVal[1])
		#print(cur_points)
		endPoints.append(cur_points[-1])
	eL = len(endPoints)
	ey_list = [endPoints[j][0] for j in range(eL)]
	ez_list = [endPoints[j][1] for j in range(eL)]
	plt.scatter(ey_list, ez_list, s = 1, color='blue')
	#plt.scatter(k_py_list, k_pz_list, s = 1, color='lightgreen')
	plt.scatter(py_list, pz_list, s = 1, color='lightgreen')
	plt.show()
