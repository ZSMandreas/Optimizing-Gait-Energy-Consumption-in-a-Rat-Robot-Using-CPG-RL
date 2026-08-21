import pygame


class joy_controller(object):
	"""docstring for joy_controller"""
	def __init__(self):
		super(joy_controller, self).__init__()
		pygame.init()
		pygame.joystick.init()

		self.joystick = pygame.joystick.Joystick(0)
		self.joystick.init()
		clock = pygame.time.Clock()
		self.buttons_data = [0]*4
		self.rocker_val = [0]*4
		self.arrow_key = [0]*2
	def reset(self):
		self.joystick.init()
		self.buttons_data = [0]*4
		self.rocker_val = [0]*4
		self.arrow_key = [0]*2

	def runStep(self):
		for event_ in pygame.event.get():
			if event_.type == pygame.QUIT:
				self.shutdown()
			elif event_.type ==pygame.JOYBUTTONDOWN or event_.type ==pygame.JOYBUTTONUP:
				'''
				Label Buttons ->
				"X": 0 		Circle: 1 	Triangle: 2
				Square: 3 	L1: 4 		R1: 5
				'''
				self.buttons_data[0] = self.joystick.get_button(0)
				self.buttons_data[1] = self.joystick.get_button(1)
				self.buttons_data[2] = self.joystick.get_button(2)
				self.buttons_data[3] = self.joystick.get_button(3)
			elif event_.type == pygame.JOYAXISMOTION:
				'''
				0: Left [LEFT, RIGHT] [-1, 1] 	1: Left [UP, DOWN] [-1, 1]
				2:
				3: Left [LEFT, RIGHT] [-1, 1] 	4: Left [UP, DOWN] [-1, 1]
				5:
				'''
				self.rocker_val[0] = self.joystick.get_axis(0)
				self.rocker_val[1] = self.joystick.get_axis(1)
				self.rocker_val[2] = self.joystick.get_axis(3)
				self.rocker_val[3] = self.joystick.get_axis(4)
			elif event_.type == pygame.JOYHATMOTION:
				'''
				Orientation Buttons ->
				[-1, 0]: left 		[1, 0]: right
				[0, -1]: down 		[0, 1]: up
				'''
				self.arrow_key = self.joystick.get_hat(0)
		return self.buttons_data, self.rocker_val, self.arrow_key	

	def shutdown(self):
		pygame.quit()
		
