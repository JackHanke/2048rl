

# the agent that learns nothing an acts randomly
class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'
        self.type = 'offline'
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = simple_exponent_state_rep
    def choose(self, state, invalid_moves): return random.choice((0, 1, 2, 3))
    def update(self): pass
