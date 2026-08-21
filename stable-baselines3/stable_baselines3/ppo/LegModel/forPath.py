import math
import numpy as np

class LegPath(object):
	"""docstring for ForeLegPath"""
	def __init__(self, pathType="circle"):
		super(LegPath, self).__init__()
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
		self.para_FU = [[-0.00, -0.045], [0.03, 0.01]]
		self.para_FD = [[-0.00, -0.045], [0.03, 0.005]]
		self.para_HU = [[-0.005, -0.05], [0.03, 0.01]]
		self.para_HD = [[-0.005, -0.05], [0.03, 0.005]]
		


	def getOvalPathPoint(self, radian, leg_name, halfPeriod, action):
		"""
		输入为统一 action（4条腿参数拼在一起的16维向量），从中提取当前腿的参数用于生成椭圆轨迹。
		"""
		# print(idx)
		ox_offset, oy_offset, rx, ry = action
		# print(ox_offset)
		# print(oy_offset)
  
		base_origin = {
		    "LF": [ 0.2,  0.1],
		    "RF": [ 0.2, -0.1],
		    "LH": [-0.2,  0.1],
		    "RH": [-0.2, -0.1]
		}[leg_name]
	
		# base_origin = self.default_origin[leg_flag]
		originPoint = [base_origin[0] + ox_offset, base_origin[1] + oy_offset]
	
		# 当前轨迹角度（归一化 radian）
		if radian < halfPeriod * math.pi:
			cur_radian = radian / halfPeriod
		else:
			cur_radian = radian / (2 - halfPeriod)
	
		# 椭圆轨迹点
		trg_x = originPoint[0] + rx * math.cos(cur_radian)
		trg_y = originPoint[1] + ry * math.sin(cur_radian)
	
		return [trg_x, trg_y]