import random
from functions.repfuncs import simple_exponent_state_rep

# the agent that learns nothing an acts randomly
class RandomAgent:
    def __init__(self):
        self.name = 'RandomAgent'
        self.type = 'offline'
        self.version_num = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = simple_exponent_state_rep
    def choose(self, **kwargs): return random.choice((0, 1, 2, 3))
    def update(self, **kwargs): pass
