class CSpinePos(object):
	"""docstring for CSpinePos"""
	def __init__(self, spine=0, tail=0):
		super(CSpinePos, self).__init__()
		self.spine = spine
		self.tail = tail


class CSpine(object):
	"""docstring for CSpine"""
	_RangeLeft = 40
	_RangeRight = 40
	_posStreched = 120
	_posCrouched = 180
	_cOffsetSpine = 0
	_cOffsetTail = 0
	_cOffsetFlex = 0
	_posCentre = 180
	_spineStep = 10
	_posSpineCentre = _posCentre + _cOffsetSpine
	_posTailCentre = _posCentre + _cOffsetTail
	_posFarLeft = _posSpineCentre - _RangeLeft
	_posFarRight = _posSpineCentre + _RangeRight
	_posTailFarLeft = _posTailCentre - _RangeLeft
	_posTailFarRight = _posTailCentre + _RangeRight

	_dir = True
	_leftStart = True
	_rightStart = True
	_rightTailStart = True
	_leftTailStart = True
	_curSP = _posCentre + _cOffsetSpine
	_curTL = _posCentre + _cOffsetTail

	_leftStepsize = 0
	_rightStepsize = 0
	_TailStepsize = 0
	def __init__(self):
		super(CSpine).__init__()

	def stretch(self):	# returns a stretched position value
		return self._posStreched
	def crouch(self):	# returns a crouched position value
		return self._posCrouched
	
	def moveTailLeft(self, length):			# move right centre to 50 and back
		if self._leftTailStart:
			# set stepsize to got there and back again
			self._TailStepsize = float(self._RangeLeft) / (float(length) / 2)
			self._leftTailStart = False
			self._dir = True
			self._curTL = self._posTailCentre
			return self._curTL
		# set new positions of the spine
		if self._dir and (self._curTL > self._posTailFarLeft):
			self._curTL = self._curTL - self._TailStepsize	# go left
		elif not self._dir and (self._curTL < self._posTailCentre):
			self._curTL = self._curTL + self._TailStepsize	# go centre
		else:
			# check wether motion completed
			if self._curTL >= self._posTailCentre:
				self._curTL = self._posTailCentre
				self._leftTailStart = True
				return self._curTL
			self._dir = not self._dir		# change direction
		return self._curTL;
	
	def moveTailRight(self, length):		# move right centre to 130 and back
		if self._rightTailStart:
			#set stepsize to got there and back again
			self._TailStepsize = float(self._RangeRight) / (float(length) / 2)
			self._rightTailStart = False
			self._dir = True
			self._curTL = self._posTailCentre
			return self._curTL

		# set new positions of the spine
		if self._dir and (self._curTL < self._posTailFarRight):
			self._curTL = self._curTL + self._TailStepsize	# go right
		elif not self._dir and (self._curTL > self._posTailCentre):
			self._curTL = self._curTL - self._TailStepsize	# go centre
		else:
			# check wether motion completed
			if self._curTL <= self._posTailCentre:
				self._curTL = self._posTailCentre
				self._rightTailStart = True
				return self._curTL
			self._dir = not self._dir 		# change direction
		return self._curTL;

	# moving the Spine and Tail smoothly during walking
	# every call to this funtion gives the next value for a given amount of iterations (length)
	def moveLeft(self, length):
		if self._leftStart:
			self._leftStepsize = length / self._posFarLeft
			self._leftStart = False
		self._curSP = self._curSP + self._leftStepsize
		self._curTL = self._curTL + self._leftStepsize
		if self._curSP > self._posFarLeft:
			self._curSP = self._posFarLeft
			self._leftStart = True
		if self._curTL > self._posFarLeft:
			self._curTL = self._posFarLeft
			self._leftStart = True
		return CSpinePos(self._curSP, self._curTL)

	def moveRight(self, length):
		if self._rightStart:
			self._rightStepsize = length / self._posFarRight
			self._rightStart = False
		self._curSP = self._curSP + self._rightStepsize
		self._curTL = self._curTL + self._rightStepsize
		if self._curSP > self._posFarRight:
			self._curSP = self._posFarRight
			self._rightStart = True
		if self._curTL > self._posFarRight:
			self._curTL = self._posFarRight
			self._rightStart = True
		return CSpinePos(self._curSP, self._curTL)

	# moving the Spine and Tail stepwise for direction control
	# every call to this funtion gives the next hihger bending iteration
	# until the maximum bending is reached.
	def moveStepLeft(self, length):
		self._curSP = self._curSP - self._spineStep
		if self._curSP < self._posFarLeft:
			self._curSP = self._posFarLeft
		return CSpinePos(self._curSP, self._curTL)
	def moveStepRight(self, length):
		self._curSP = self._curSP + self._spineStep
		if self._curSP > self._posFarRight:
			self._curSP = self._posFarRight;
		return CSpinePos(self._curSP, self._curTL);
	def centre(self):
		return CSpinePos((self._posCentre + self._cOffsetSpine), (self._posCentre + self._cOffsetTail))
		