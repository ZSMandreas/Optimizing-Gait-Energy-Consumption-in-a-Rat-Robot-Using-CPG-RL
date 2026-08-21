import math
import numpy as np

class LegPath(object):
	"""docstring for ForeLegPath"""
	def __init__(self, pathType="circle", b_scale=1.0, a_scale=1.0, omega_profile=None):
		super(LegPath, self).__init__()
		self.path_type = "ellipse"
		self.omega_profile = omega_profile
		self.shape_params = {
			"y_family": "Y_B_quintic",
			"z_family": "Z_B_quartic",
			"y_k": 0.2,
			"z_k": 0.2,
		}
		'''
		self.para_FU = [[-0.005, -0.05], [0.03, 0.01]]
		self.para_FD = [[-0.005, -0.05], [0.03, 0.005]]
		self.para_HU = [[0.00, -0.05], [0.03, 0.01]]
		self.para_HD = [[0.00, -0.05], [0.03, 0.005]]
		'''
		#self.para_FU = [[-0.005, -0.045], [0.02, 0.01]]
		#self.para_FD = [[-0.005, -0.045], [0.02, 0.005]]
		#self.para_HU = [[-0.005, -0.054], [0.03, 0.01]]
		#self.para_HD = [[-0.005, -0.054], [0.03, 0.005]]
		"""
		b_scale：垂直振幅缩放（同时作用于 b_up 和 b_dn）
		a_scale：水平振幅缩放（作用于 a）
		"""
		b_up = 0.010 * b_scale
		b_dn = 0.005 * b_scale
		a = 0.030 * a_scale

		self.para_FU = [[-0.00, -0.045], [a, b_up]]
		self.para_FD = [[-0.00, -0.045], [a, b_dn]]
		self.para_HU = [[-0.005, -0.05], [a, b_up]]
		self.para_HD = [[-0.005, -0.05], [a, b_dn]]
		self.set_shape_params(path_type=pathType)
		
	def set_omega_profile(self, omega_profile):
		"""Set optional omega(phi) profile, None keeps constant-step behavior."""
		self.omega_profile = omega_profile

	def advance_step(self, cur_step, step_num, dt):
		"""Advance shared phase step by either default +1 or omega-modulated increment."""
		if step_num <= 0:
			return cur_step
		if self.omega_profile is None:
			return (cur_step + 1.0) % float(step_num)
		phi = 2.0 * math.pi * (float(cur_step) % float(step_num)) / float(step_num)
		omega = float(self.omega_profile(phi))
		# Convert dphi to dstep under shared-base-step semantics.
		dstep = omega * float(dt) * float(step_num) / (2.0 * math.pi)
		return (float(cur_step) + dstep) % float(step_num)



	def getOvalPathPoint(self, radian, leg_flag, halfPeriod):
		pathParameter = None
		cur_radian = 0
		if leg_flag == "F":
			if radian < halfPeriod*math.pi:
				pathParameter = self.para_FU
				cur_radian = radian/halfPeriod
			else:
				pathParameter = self.para_FD
				cur_radian = (radian)/(2-halfPeriod)
		else:
			if radian < halfPeriod*math.pi:
				pathParameter = self.para_HU
				cur_radian = radian/halfPeriod
			else:
				pathParameter = self.para_HD 
				cur_radian = (radian)/(2-halfPeriod) 

		originPoint = pathParameter[0] # 椭圆中心 (x0, y0)
		ovalRadius = pathParameter[1] # 椭圆半径 (a, b)

		trg_x = originPoint[0] + ovalRadius[0] *math.cos(cur_radian)
		trg_y = originPoint[1] + ovalRadius[1] *math.sin(cur_radian)
		return [trg_x, trg_y]

	def set_shape_params(self, path_type=None, shape_params=None):
		"""Update runtime trajectory family without touching ellipse parameters."""
		if path_type is not None:
			pt = str(path_type).lower()
			if pt in ("circle", "oval", "ellipse"):
				self.path_type = "ellipse"
			elif pt in ("poly", "v3", "custom"):
				self.path_type = "poly"
		if shape_params is not None:
			merged = dict(self.shape_params)
			merged.update(shape_params)
			self.shape_params = merged

	def _y_shape(self, s):
		fam = self.shape_params.get("y_family", "Y_B_quintic")
		k = float(self.shape_params.get("y_k", 0.2))
		alpha = float(self.shape_params.get("y_alpha", 0.0))
		k = min(max(k, 1e-4), 0.49)
		if fam == "Y_A_cosine":
			return math.cos(math.pi * s)
		if fam.startswith("Y_C_trap"):
			if s < k:
				return 1.0 - (s * s) / (k * (1.0 - k))
			if s < 1.0 - k:
				return 1.0 - (2.0 * s - k) / (1.0 - k)
			t = 1.0 - s
			return -1.0 + (t * t) / (k * (1.0 - k))
		if fam in ("Y_D_bell", "Y_D_bell_velocity"):
			return 1.0 - 2.0 * (s / 2.0 - math.sin(2.0 * math.pi * s) / (4.0 * math.pi))
		if fam == "Y_E_septic_alpha":
			# Septic perturbation on top of quintic base with bounded apex shift
			p5 = 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
			q7 = s**3 * (1.0 - s)**3 * (1.0 - 2.0 * s)
			return 1.0 - 2.0 * p5 + alpha * q7 * 32.0
		# Default: Y_B_quintic
		p5 = 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
		return 1.0 - 2.0 * p5

	def _z_shape(self, s):
		fam = self.shape_params.get("z_family", "Z_B_quartic")
		k = float(self.shape_params.get("z_k", 0.2))
		k = min(max(k, 1e-4), 0.49)
		if fam == "Z_A_sine":
			return math.sin(math.pi * s)
		if fam == "Z_C_sextic":
			return 64.0 * s**3 * (1.0 - s)**3
		if fam.startswith("Z_D_trap"):
			if s < k:
				return s / k
			if s < 1.0 - k:
				return 1.0
			return (1.0 - s) / k
		if fam == "Z_E_sin_cubed":
			ss = math.sin(math.pi * s)
			return ss * ss * ss
		# Default: Z_B_quartic
		return 16.0 * s**2 * (1.0 - s)**2

	def getPathPoint(self, radian, leg_flag, halfPeriod):
		"""Unified interface: keep ellipse default, optional polynomial family."""
		if self.path_type != "poly":
			return self.getOvalPathPoint(radian, leg_flag, halfPeriod)
		pathParameter = None
		cur_radian = 0.0
		is_swing = False
		if leg_flag == "F":
			if radian < halfPeriod * math.pi:
				pathParameter = self.para_FU
				cur_radian = radian / halfPeriod
				is_swing = True
			else:
				pathParameter = self.para_FD
				cur_radian = radian / (2 - halfPeriod)
		else:
			if radian < halfPeriod * math.pi:
				pathParameter = self.para_HU
				cur_radian = radian / halfPeriod
				is_swing = True
			else:
				pathParameter = self.para_HD
				cur_radian = radian / (2 - halfPeriod)
		originPoint = pathParameter[0]
		ovalRadius = pathParameter[1]
		if not is_swing:
			return [
				originPoint[0] + ovalRadius[0] * math.cos(cur_radian),
				originPoint[1] + ovalRadius[1] * math.sin(cur_radian),
			]
		s = min(max(cur_radian / math.pi, 0.0), 1.0)
		trg_x = originPoint[0] + ovalRadius[0] * self._y_shape(s)
		trg_y = originPoint[1] + ovalRadius[1] * self._z_shape(s)
		return [trg_x, trg_y]
