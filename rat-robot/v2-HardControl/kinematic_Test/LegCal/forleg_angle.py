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

	def angel_2_pos(self, q1, q2):
		Ey = self.len[0]*math.sin(q1)
		Ez = 0 - self.len[0]*math.cos(q1)

		Cy = 0 - self.len[2]*math.sin(q2)
		Cz = self.len[1] + self.len[2]*math.cos(q2)


		CE = math.sqrt((Ey-Cy)*(Ey-Cy) + (Ez-Cz)*(Ez-Cz))
		a_ECz = math.acos((Cz-Ez)/CE) * (Ey)/abs(Ey)
		a_ECD = self.LawOfCosines_angle(CE, self.len[3], self.len[4])
		if a_ECD == -10:
			return []
		a_DCz = a_ECD + a_ECz

		'''
		Dy = Cy + self.len[3]*math.cos(a_DCz-math.pi/2)
		Dz = Cz + self.len[3]*math.sin(a_DCz-math.pi/2)
		print(math.sqrt((Cy-Dy)*(Cy-Dy)+(Cz-Dz)*(Cz-Dz)),
			' -- ', math.sqrt((Ey-Dy)*(Ey-Dy)+(Ez-Dz)*(Ez-Dz)))
		'''
		Dy = Cy + self.len[3]*math.sin(a_DCz)
		Dz = Cz - self.len[3]*math.cos(a_DCz)
		

		DE_y = Dy - Ey
		DE_z = Dz - Ez

		FE_DE = self.len[5]/self.len[4]
		Fy = Ey - FE_DE*DE_y
		Fz = Ez - FE_DE*DE_z


		AF = math.sqrt((Fy-0)*(Fy-0)+(Fz-0)*(Fz-0))
		a_AEF = self.LawOfCosines_angle(self.len[0], self.len[5], AF)
		BD = math.sqrt((self.By-Dy)*(self.By-Dy)+(self.Bz-Dz)*(self.Bz-Dz))
		a_BCD = self.LawOfCosines_angle(self.len[2], self.len[3], BD)

		a_CDE = self.LawOfCosines_angle(self.len[3], self.len[4], CE)
		print("<AEF = :", 180*a_AEF/math.pi)
		print("<BCD = :", 180*a_BCD/math.pi)
		print("<CDE = :", 180*a_CDE/math.pi)
		DF = self.len[5] + self.len[4]
		print("DF * <AEF: --> cos: ", DF*math.cos(a_AEF)/2, " | sin: ", DF*math.sin(a_AEF)/2)
		print("sub_EF * <AEF: --> cos: ", (DF/2-self.len[4])*math.cos(a_AEF), " | sin: ", (DF/2-self.len[4])*math.sin(a_AEF))
		print("EF * <AEF: --> cos: ", self.len[5]*math.cos(a_AEF), " | sin: ", self.len[5]*math.sin(a_AEF))
		print("DE * <AEF: --> cos: ", self.len[4]*math.cos(a_AEF), " | sin: ", self.len[4]*math.sin(a_AEF))
		print("CD * <BCD: --> cos: ", self.len[3]*math.cos(a_BCD)/2, " | sin: ", self.len[3]*math.sin(a_BCD)/2)
		return [[0,0], [self.By, self.Bz],[Cy, Cz],
			[Dy, Dz], [Ey, Ez], [Fy, Fz]]

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
	