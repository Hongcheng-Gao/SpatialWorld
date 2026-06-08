class StoryScore:
	def __init__(self, player_name, player_steps):
		self.player_name = player_name
		self.player_steps = player_steps

	def get_data(self):
		return "{} - {}".format(self.player_name, self.player_steps)

	def get_score_ttl(self):
		return sum(self.player_steps)
