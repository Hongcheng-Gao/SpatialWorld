class Player:
	def __init__(self, x_position, y_position):
		self.x = x_position
		self.y = y_position
		self.steps = 0
		print("Player Initialized")

	def move_up(self):
	    self.y -= 1

	def move_down(self):
		self.y += 1

	def move_left(self):
		self.x -= 1

	def move_right(self):
		self.x += 1