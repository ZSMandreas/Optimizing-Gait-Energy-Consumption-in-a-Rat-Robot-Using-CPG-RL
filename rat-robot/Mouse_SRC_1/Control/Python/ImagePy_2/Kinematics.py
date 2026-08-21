import math

# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
class CLegPos(object):
	"""docstring for CLegPos"""
	def __init__(self, leg=0, coil=0):
		super(CLegPos, self).__init__()
		self.leg = leg;
		self.coil = coil
# ---------------------------------------------------------------- #
class CKoord(object):
	"""docstring for CKoord"""
	def __init__(self, px=0.0, py=0.0):
		super(CKoord, self).__init__()
		self.x = px
		self.y = py
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #

ServosFlag = "NRPServos"
class CKinematics(object):
	"""docstring for CKinematics"""
	#Kinematic Valuess////////////////////////////////////////////////////////////////////
	_rp1 = 7.5 												#//radius of the pulley
	#Foreleg---------------------------------------------------------------------------
	_fl1 = 27.8												#l_humerus length upper arm
	_fl2 = 26.4												#l_ulna length forearm
	_fl3 = 14.26											#l_hand length hand
	#initial Angles at x=y=0
	_ft1i = 54.24											#180° - init angle shoulder
	_ft2i = 87.65											#180° - init angle elbow
	_ft3i = 155.0											#init angle hand
	#Foreleg_String_Calculation
	_fltibia = 10											#string attachment point on tibia
	_frhip = 4												#effective radius of the pulley
	_fahip = 116.79											#initial angle of string routed on the hip
	_flipfree = 41.198										#inital length of free string Attachment-Hip
	_flsi = _flipfree + (_fahip * 3.1415 / 180 * _frhip)	#initial calculated length
	#Hindleg---------------------------------------------------------------------------
	_hl1 = 35.0	#l_femur 	(thigh Oberschenkel)
	_hl2 = 39.4	#l_tibia 	(lower leg Unterschenkel)
	_hl3 = 20	#l_toe		(toe Zeh)
	#initial Angles at x=y=0
	_ht1i = 7.49	#a_femur	(init angle femur)
	_ht2i = 108.93	#a_knee	(180 - init angle knee)
	_ht3i = 110.0	#a_toe		(init angle toe)
	#Hindleg_String_Calculation
	_hltibia = 17.4											#string attachment point on tibia
	_hrhip = 4												#effective radius of the pulley
	_hahip = 160.26 										#initial angle of string routed on the hip
	_hlhipfree = 37.687										#initial length of string hip-attachment
	_hlsi = _hlhipfree + (_hahip * 3.1415 / 180 * _hrhip)	#initial total length
	#--------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------
	flOffset = 0		#offset value for foreright hip servo
	frOffset = 0		#offset value for foreright hip servo
	hlOffset = 0		#offset value for hindleft hip servo
	hrOffset = 0		#offset value for hindright hip servo
	def __init__(self):
		super(CKinematics, self).__init__()
		if ServosFlag == "NRPServos":
			self.lhlHipInit = 180			#ind left hip
			self.rhlHipInit = 180			#hind right hip
			self.lhlCoilInit = 180			#hind left Coil
			self.rhlCoilInit = 180			#hind right Coil
			#TODO save initControlled positions here
			#Forelegs----------------------------------------
			self.lflHipInit = 180			#fore left Hip
			self.rflHipInit = 180			#fore right Hip
			self.lflCoilInit = 180			#fore left Coil
			self.rflCoilInit = 180			#fore right coil
		else:
			self.lhlHipInit = 150			#ind left hip
			self.rhlHipInit = 30			#hind right hip
			self.lhlCoilInit = 40			#hind left Coil
			self.rhlCoilInit = 40			#hind right Coil
			#TODO save initControlled positions here
			#Forelegs----------------------------------------
			self.lflHipInit = 90			#fore left Hip
			self.rflHipInit = 90			#fore right Hip
			self.lflCoilInit = 80			#fore left Coil
			self.rflCoilInit = 80			#fore right coil

	#Control Functions////////////////////////////////////////////////////////////////////
	def ikforeleg(self, fx, fy, side):
		# side can be l( left leg ) and r (right leg)
		# ik of the fore leg, returns the servoangles required for IK
		# theta3 (hand angle) is assumed to be a constant to prevent complex calculations
		# Assuming the kinematics only, no ground contact forces

		#---- setup Values -----------------------------------
		# Initial position of leg in taskspace
		# (The Toe is located 28 in Y Direction and 55.15 in X DIrection)
		fpos_init= [28, 55.15]		# corresponding to trot gait
		#---- intermediate Values -----------------------
		ft1 = 0		# shoulder angle
		ft2 = 0		# elbow angle
		flb = 0		# /length elbow-fingertip
		ftb = 0		# angle upper_arm - flb
		# shifting the goal coordinate by the initial toe coordinate
		# origin position of the leg (easier for trajectory planning)(relative to center point of trajectory circle)
		fx = fx - fpos_init[0]
		fy = fy - fpos_init[1]
		#---- IK Calculations -----------------------
		flb = math.sqrt(pow(self._fl3, 2) + pow(self._fl2, 2) -\
			(2 * self._fl2 * self._fl3 * math.cos(self._ft3i * math.pi / 180)))
		ftb = math.acos(((pow(fx, 2) + pow(fy, 2)) -\
			(pow(self._fl1, 2) + pow(flb, 2))) / (-2 * self._fl1 * flb));
		ft2 = ftb + math.asin(self._fl3 * math.sin(self._ft3i * math.pi / 180) / flb)

		# differentiate between X coordinate -> passing of horizontal line for Shoulder Angle
		if fx < 0:
			ft1 = math.pi - math.atan(fy / fx) - math.asin(flb * math.sin(ftb) / math.sqrt(pow(fx, 2) + pow(fy, 2)))
		elif fx > 0:
			ft1 = - math.atan(fy / fx) - math.asin(flb * math.sin(ftb) / math.sqrt(pow(fx, 2) + pow(fy, 2)))
		else:
			ft1 = (90 * math.pi / 180) - math.asin(flb * math.sin(ftb) / math.sqrt(pow(fx, 2) + pow(fy, 2)))

		#Rückrechnung in Grad
		ft1 = ft1 * 180 / math.pi
		ft2 = ft2 * 180 / math.pi
		return CLegPos(self.foreHipServoAngle(ft1, side), self.foreKneeServoAngle(ft1, ft2, side))

	def ikhindleg(self, hx, hy, side):
		# side can be l( left leg ) and r (right leg)
		# ik of the hind leg, returns the servoangles required for IK
		# theta3 (foot angle) is assumed to be a constant to prevent complex calculations
		# The string will be routed on the tibia for now.
		# Assuming the kinematics only, no ground contact forces

		# ---- setup Values -----
		# Initial position of leg in taskspace
		# (The Toe is located 20mm in Y Direction and 59.15mm in X DIrection)
		hpos_init = [20, 59.15]
		# ---- intermediate Values -----
		ht1 = 0		# Hip Angle - theta 1
		ht2 = 0		# Knee Angle - theta 2
		hlb = 0		# length knee-toe
		htb = 0		# angle oberschenkel-knee-toe
		# shifting the goal coordinate by the initial toe coordinate
		hx = hx - hpos_init[0]
		hy = hy - hpos_init[1]
		# ---- IK Calculations -----------------------
		hlb = math.sqrt(pow(self._hl3, 2) + pow(self._hl2, 2) -\
			(2 * self._hl2 * self._hl3 * math.cos(self._ht3i * math.pi / 180)));
		htb = math.acos(((pow(hx, 2) + pow(hy, 2)) -\
			(pow(self._hl1, 2) + pow(hlb, 2))) / (-2 * self._hl1 * hlb));
		ht2 = htb - math.asin(self._hl3 * math.sin(self._ht3i * math.pi / 180) / hlb);

		# differentiate between X coordinate -> passing of horizontal line for Hip Angle
		if hx > 0:
			ht1 = math.pi + math.atan(hy / hx) - math.asin(hlb * math.sin(htb) / math.sqrt(pow(hx, 2) + pow(hy, 2)))
		elif hx < 0:
			ht1 = math.atan(hy / hx) - math.asin(hlb * math.sin(htb) / math.sqrt(pow(hx, 2) + pow(hy, 2)))
		else:
			ht1 = (90 * math.pi / 180) - math.asin(hlb * math.sin(htb) / math.sqrt(pow(hx, 2) + pow(hy, 2)))

		ht1 = ht1 * 180 / math.pi
		ht2 = ht2 * 180 / math.pi

		return CLegPos(self.hindHipServoAngle(ht1, side), self.hindKneeServoAngle(ht1, ht2, side))

	#Angle Calculation////////////////////////////////////////////////////////////////////
	#Foreleg
	def foreKneeServoAngle(self, ft1, ft2, side):
		lhip = 0;
		#calcuate length of string at the hip turn
		if ft1 > self._ft1i:		# when leg is moved forward
			lhip = (self._fahip - (ft1 - self._ft1i)) * math.pi / 180 * 3.7		# the string length at the hip pulley,
		else:						# when leg is moved backward
			lhip = (self._fahip + (self._ft1i - ft1)) * math.pi / 180 * 3.7
		#calculate length of string from hip to attachment point at leg
		centerdist = math.sqrt(pow(self._fl1, 2) + pow(self._fltibia, 2) -\
			(2 * self._fl1 * self._fltibia * math.cos(ft2 * math.pi / 180)))	# length from attachment point to hip centre
		lknee = math.sqrt(pow(centerdist, 2) - pow(self._frhip, 2))				# length from hip tanget to attachment point

		fls = lhip + lknee			# total calculated length

		# calculation for left/right side
		# *180/PI is for change from radians to degrees
		if side == 'l':			#left side-----------------
			if fls > self._flsi:		# need to release string, turn cw (+)
				return self.lflCoilInit + (self.kneeservoangle(fls - self._flsi)) * 180 / math.pi
			else:				# need to tension string
				return self.lflCoilInit - (self.kneeservoangle(self._flsi - fls)) * 180 / math.pi
		else:					# right side-------------------
			if fls > self._flsi:		# need to release string, turn anticw(-)
				return self.rflCoilInit - (self.kneeservoangle(fls - self._flsi)) * 180 / math.pi
			else:				# need to tension string, turn cw (+)
				return self.rflCoilInit + (self.kneeservoangle(self._flsi - fls)) * 180 / math.pi
	
	def foreHipServoAngle(self, ft1, side):
		if side == 'l':
			if ft1 > self._ft1i:		# move leg forward, turn cw (+)
				return self.lflHipInit + (ft1 - self._ft1i) + self.flOffset
			else:					# move leg back
				return self.lflHipInit - (self._ft1i - ft1) + self.flOffset
		else:
			if ft1 > self._ft1i:			# move leg forward, turn anticw (-)
				return self.rflHipInit - (ft1 - self._ft1i) + self.frOffset
			else:					# move leg back,
				return self.rflHipInit + (self._ft1i - ft1) + self.frOffset
	#Hindleg
	def hindKneeServoAngle(self, ht1, ht2, side):
		lhip = 0
		if ht1 > self._ht1i:		#when leg is moved backward
			lhip = (self._hahip - (ht1 - self._ht1i)) * math.pi / 180 * 3.7	#the string length at the hip pulley,
		else:						#when leg is moved forward
			lhip = (self._hahip + (self._ht1i - ht1)) * math.pi / 180 * 3.7

		# calculate length of string from hip to attachment point at leg
		centerdist = math.sqrt(pow(self._hl1, 2) + pow(self._hltibia, 2) - \
			(2 * self._hl1 * self._hltibia * math.cos(ht2 * math.pi / 180)))#length from attachment point to hip centre
		lknee = math.sqrt(pow(centerdist, 2) - pow(self._hrhip, 2))			#length from hip tanget to attachment point
		hls = lhip + lknee													#total calculated length
		# calculation for left/right side
		# *180/PI is for change from radians to degrees
		if side == 'l':				#left side-----------------
			if hls > self._hlsi:		# need to release string, turn anticw (-)
				return self.lhlCoilInit - (self.kneeservoangle(hls - self._hlsi)) * 180 / math.pi
			else:					# need to tension string
				return self.lhlCoilInit + (self.kneeservoangle(self._hlsi - hls)) * 180 / math.pi
		else:						#right side-----------------
			if hls > self._hlsi:			#need to release string, turn cw(+)
				return self.rhlCoilInit + (self.kneeservoangle(hls - self._hlsi)) * 180 / math.pi
			else:					#need to tension string
				return self.rhlCoilInit - (self.kneeservoangle(self._hlsi - hls)) * 180 / math.pi

	def hindHipServoAngle(self, ht1, side):
		if side == 'l':
			if ht1 > self._ht1i:		# move leg back,  turn anti cw (-)
				return self.lhlHipInit - (ht1 - self._ht1i) + self.hlOffset
			else:						# move leg forward
				return self.lhlHipInit + (self._ht1i - ht1) + self.hlOffset
		else:
			if ht1 > self._ht1i:		# move leg back,  turn cw (+)
				return self.rhlHipInit + (ht1 - self._ht1i) + self.hrOffset
			else:						# move leg forward
				return self.rhlHipInit - (self._ht1i - ht1) + self.hrOffset

	#
	def kneeservoangle(self, l):
		return l / self._rp1