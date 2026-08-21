import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint,Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import shapely.geometry as geometry
from shapely.ops import nearest_points



# ---------------- utility functions ----------------
def law_of_cosines_angle(la, lb, lc):
	cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb)
	if abs(cos_val) > 1:
		return -10
	return math.acos(cos_val)

def check_cross(line1, line2):
	C, D = line1
	A, E = line2
	area_CDA = (C[0]-A[0])*(D[1]-A[1]) - (C[1]-A[1])*(D[0]-A[0])
	area_CDE = (C[0]-E[0])*(D[1]-E[1]) - (C[1]-E[1])*(D[0]-E[0])
	area_AEC = (A[0]-C[0])*(E[1]-C[1]) - (A[1]-C[1])*(E[0]-C[0])
	area_AED = (A[0]-D[0])*(E[1]-D[1]) - (A[1]-D[1])*(E[0]-D[0])
	if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
		return []
	tmp = area_AEC / (area_CDE - area_CDA)
	dx = tmp * (D[0] - C[0])
	dy = tmp * (D[1] - C[1])
	return [C[0] + dx, C[1] + dy]

def alpha_shape(pts, alpha):
	if len(pts) < 4:
		return MultiPoint(list(pts)).convex_hull
	tri = Delaunay(pts)
	edges = set()
	for ia, ib, ic in tri.simplices:
		pa, pb, pc = pts[ia], pts[ib], pts[ic]
		a = np.linalg.norm(pb - pc)
		b = np.linalg.norm(pa - pc)
		c = np.linalg.norm(pa - pb)
		area = 0.5 * abs(np.cross(pb - pa, pc - pa))
		R = a*b*c / (4.0*area + 1e-12)       # 外接圆半径
		if R < 1.0 / alpha:                  # alpha 判据
			edges.update([(ia, ib), (ib, ic), (ic, ia)])
	edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
	m = unary_union(edge_segments)
	return unary_union(list(polygonize(m)))



# ---------------- leg model with CBz limit ----------------
class WorkspaceDetection:
	def __init__(self, leg_params):
		self.len = leg_params
		self.By = 0.0
		self.Bz = self.len[1]
		self.limit_CBz = law_of_cosines_angle(self.len[2], self.len[1], 0.0075)
		self.limit_DCB = law_of_cosines_angle(0.012735, self.len[2], 0.002)
		self.limit_AEF = law_of_cosines_angle(0.01025, 0.01025, 0.0042)


	def angel_2_pos(self, q1, q2):
		PI = math.pi
		Ey = self.len[0]*math.cos(q1)
		Ez = self.len[0]*math.sin(q1)

		Cy = -self.len[2]*math.sin(q2)
		Cz = self.len[1] + self.len[2]*math.cos(q2)

		CE = math.hypot(Ey - Cy, Ez - Cz)
		if CE == 0:
			return []

		a_ECz = math.acos((Cz - Ez)/CE) * np.sign(Ey - Cy)
		a_ECD = law_of_cosines_angle(CE, self.len[3], self.len[4])
		if a_ECD == -10:
			return []
		a_DCz = a_ECD + a_ECz

		Dy = Cy + self.len[3]*math.sin(a_DCz)
		Dz = Cz - self.len[3]*math.cos(a_DCz)

		DEy, DEz = Dy - Ey, Dz - Ez
		Fy = Ey - (self.len[5]/self.len[4]) * DEy
		Fz = Ez - (self.len[5]/self.len[4]) * DEz

		BD = math.hypot(self.By - Dy, self.Bz - Dz)

		# # CBz angle limit
		# Bz = self.len[1]
		# angle_CBz = math.acos((Cz - Bz) / self.len[2])
		# # angle_CBz = -q2
		# if angle_CBz == -10 or angle_CBz > self.limit_CBz:
		# # if q2 == -10 or q2 > self.limit_CBz:
		# 	return []
		

		# cross check
		cross = check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
		if cross:
			dist = math.hypot(cross[0]-Ey, cross[1]-Ez)
			if dist < self.len[0]*1.5:
				return []

		# angle range checks (30°–150°) for AEF and BCD
		AF = math.hypot(Fy, Fz)
		a_AEF = law_of_cosines_angle(self.len[0], self.len[5], AF)
		a_BCD = law_of_cosines_angle(self.len[2], self.len[3], BD)
		if (a_AEF < PI/6) or (a_AEF > PI*5/6) or (a_BCD < PI/6) or (a_BCD > PI*5/6):
		# if a_AEF > self.limit_AEF ora_BCD > self.limit_DCB: 
			return []

		return Fy, Fz
	
	

# ----------- parameters and sampling grid ------------
leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
model = WorkspaceDetection(leg_params)

grid = 400
q_vals = np.linspace(-3.0, 3.0, grid)
Fy_col, Fz_col = [], []

for q1 in q_vals:
	for q2 in q_vals:
		res = model.angel_2_pos(q1, q2)
		if res:
			if res[1] < 0:
				Fy_col.append(res[0])
				Fz_col.append(res[1])

Fy_arr = np.array(Fy_col)
Fz_arr = np.array(Fz_col)
points = np.vstack((Fy_arr, Fz_arr)).T

alpha = 100        # 越大轮廓越紧，可按需要调
alpha_poly = alpha_shape(points, alpha)   # shapely Polygon

# -------- is_valid() 改为 α-shape 判定 ----------
safe_poly = alpha_poly.buffer(1e-6)

def is_valid(Fy, Fz):
	return safe_poly.covers(Point(Fy, Fz))

def nearest_valid_point(Fy, Fz):
	"""将非法点投影到 alpha shape 的边界上"""
	point = Point(Fy, Fz)
	nearest = nearest_points(alpha_poly, point)[0]
	return nearest.x, nearest.y

Fy = 0.4
Fz = -0.1

if not is_valid(Fy,Fz):
    Fy, Fz = nearest_valid_point(Fy, Fz)  # 或直接拒绝 / 惩罚

print(Fy,Fz)
	


# plot
plt.figure(figsize=(6,6))
plt.scatter(Fy_arr, Fz_arr, s=2, c='purple', alpha=0.6)
plt.title("Workspace with base constraints + ∠CBz ≤ limit")
plt.xlabel("Fy (m)")
plt.ylabel("Fz (m)")
plt.grid(True)
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(Fy_arr, Fz_arr, s=2, alpha=0.4, label='valid samples')
if alpha_poly.geom_type == 'Polygon':
	x, y = alpha_poly.exterior.xy
	plt.fill(x, y, 'r', alpha=0.2)
	plt.plot(x, y, 'r')
elif alpha_poly.geom_type == 'MultiPolygon':
	for poly in alpha_poly.geoms:
		x, y = poly.exterior.xy
		plt.fill(x, y, 'r', alpha=0.2)
		plt.plot(x, y, 'r')
plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
plt.axis('equal'); plt.grid(True); plt.legend()
plt.title("Accurate Workspace via Alpha-Shape")
plt.show()

print(is_valid(1,2))

