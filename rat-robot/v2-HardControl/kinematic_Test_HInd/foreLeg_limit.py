import math

class ForeLegM(object):
	"""docstring for ForeLegM"""
	def __init__(self, leg_params):
		super(ForeLegM, self).__init__()
		# 0: AE 	1: AB
		# 2: BC 	3: CD
		# 4: DE 	5: EF (DE+EF ~= DF)
		self.len = leg_params
		self.By = 0
		self.Bz = self.len[1]

		PI = math.pi
		self.limit_CBz = self.LawOfCosines_angle(self.len[2], self.len[1], 0.0075)
		self.limit_DCB = self.LawOfCosines_angle(0.012735, self.len[2], 0.002)
		self.limit_AEF = self.LawOfCosines_angle(0.01025, 0.01025, 0.0042)

		
		print(self.limit_CBz, ' -- ', PI/6)
		print(self.limit_DCB, ' -- ', PI/72)
		print(self.limit_AEF, ' -- ', PI/18)

	def check_cross(self, line1, line2):
		#	Link : https://zhuanlan.zhihu.com/p/158533421
		C = line1[0]
		D = line1[1]
		A = line2[0]
		E = line2[1]
		EPSINON = 1e-9
		area_CDA = (C[0]-A[0])*(D[1]-A[1]) - \
			(C[1]-A[1])*(D[0]-A[0])
		area_CDE = (C[0]-E[0])*(D[1]-E[1]) - \
			(C[1]-E[1])*(D[0]-E[0])

		area_AEC = (A[0]-C[0])*(E[1]-C[1]) - \
			(A[1]-C[1])*(E[0]-C[0])
		area_AED = (A[0]-D[0])*(E[1]-D[1]) - \
			(A[1]-D[1])*(E[0]-D[0])
		#if area_CDA*area_CDE >= -EPSINON or area_AEC*area_AED >= -EPSINON:
		#	return []

		tmp = area_AEC/(area_CDE-area_CDA)
		dx = tmp*(D[0] - C[0])
		dy = tmp*(D[1] - C[1])
		trg_x = C[0] + dx
		trg_y = C[1] + dy
		return [trg_x, trg_y]

	def angel_2_pos(self, q1, q2):
		PI = math.pi
		hip_angle = q1+ PI
		Ey = self.len[0]*math.cos(hip_angle)
		Ez = self.len[0]*math.sin(hip_angle)

		Cy = 0 - self.len[2]*math.sin(q2)
		Cz = self.len[1] + self.len[2]*math.cos(q2)


		CE = math.sqrt((Ey-Cy)*(Ey-Cy) + (Ez-Cz)*(Ez-Cz))
		a_ECz = 0
		if Ey != Cy:
			a_ECz = math.acos((Cz-Ez)/CE) * (Ey-Cy)/abs(Ey-Cy)
		a_ECD = self.LawOfCosines_angle(CE, self.len[3], self.len[4])
		if a_ECD == -10:
			return []
		a_DCz =  a_ECz - a_ECD

		Dy = Cy + self.len[3]*math.sin(a_DCz)
		Dz = Cz - self.len[3]*math.cos(a_DCz)

		DE_y = Dy - Ey
		DE_z = Dz - Ez

		FE_DE = self.len[5]/self.len[4]
		Fy = Ey - FE_DE*DE_y
		Fz = Ez - FE_DE*DE_z

		a_EFy = math.asin((Ez-Fz)/self.len[5])
		alpha = self.len[7] - a_EFy
		print(self.len[7]*180/math.pi, ' -- ', a_EFy*180/math.pi, ' -- ', alpha*180/math.pi)
		Gy = Fy - self.len[6]*math.cos(alpha)
		Gz = Fz - self.len[6]*math.sin(alpha)
		#-------------------------------------------_#

		AF = math.sqrt((0-Fy)*(0-Fy) + (0-Fz)*(0-Fz))
		BD = math.sqrt((self.By-Dy)*(self.By-Dy) + (self.Bz-Dz)*(self.Bz-Dz))

		a_AEF = self.LawOfCosines_angle(self.len[0], self.len[5] ,AF)
		a_BCD = self.LawOfCosines_angle(self.len[2], self.len[3] ,BD)

		PI = math.pi
		'''
		cross_CD_AE = self.check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
		if len(cross_CD_AE) > 0:
			dis_C = math.sqrt((cross_CD_AE[0]-Cy)*(cross_CD_AE[0]-Cy) + 
				(cross_CD_AE[1]-Cz)*(cross_CD_AE[1]-Cz))
			if dis_C > self.len[2]:
				return []
		'''
		cross_CD_AE = self.check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
		if len(cross_CD_AE) > 0:
			dis_C = math.sqrt((cross_CD_AE[0]-Ey)*(cross_CD_AE[0]-Ey) + 
				(cross_CD_AE[1]-Ez)*(cross_CD_AE[1]-Ez))
			if dis_C < self.len[0]*1.5:
				return []
		if a_AEF < PI/6 or a_BCD < PI/6  or a_AEF > PI*5/6  or a_BCD > PI*5/6:
		#if a_AEF < self.limit_AEF or a_AEF > PI - self.limit_AEF or a_BCD < self.limit_DCB :
			return []

		return [[0,0], [self.By, self.Bz],[Cy, Cz],
			[Dy, Dz], [Ey, Ez], [Fy, Fz], [Gy, Gz], cross_CD_AE]

	def LawOfCosines_edge(self, la, lb, angle_ab):
		lc_2 = la*la + lb*lb - 2*la*lb*math.cos(angle_ab)
		lc = math.sqrt(lc_2)
		return lc
	def LawOfCosines_angle(self, la, lb, lc):
		angle_ab_cos = (la*la + lb*lb - lc*lc)/(2*la*lb)
		#print("----> ", angle_ab_cos)
		if abs(angle_ab_cos) > 1:
			return -10
		angle_ab = math.acos(angle_ab_cos)
		return angle_ab
	