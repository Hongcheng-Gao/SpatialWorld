class TimeAttackScore:
	def __init__(self, player_name, levels):
		self.player_name = player_name
		self.levels = levels

	def get_data(self):
		return "{} - {}".format(self.player_name, self.levels)