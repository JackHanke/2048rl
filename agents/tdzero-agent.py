# from models import ArtificialNeuralNetwork
from models import LinearSoftmax

# 
class TDZeroAgent:
    def __init__(self):
        self.name = 'Temporal Difference Agent'
        self.type = 'online'
        self.lmbda = 0
        self.n_step = 1
        self.discounting_param = 1
        self.learning_rate = 0.01
        # self.state_rep_function = 
        # self.action_mask = 
        self.state_value_function = ArtificialNeuralNetwork(
            dims=(256, 100, 60, 1), \
            activation_funcs = [
                (leaky_relu, leaky_relu_prime), \
                (leaky_relu, leaky_relu_prime), \
                (relu, relu_prime) \
            ], \
            seed=1
        )

    def update(self):
        # 
        pass

    def choose(self, state):
        # loop through all (legal?) actions
        #   with each action, collect the reward and afterstate
        #   evaluate value of afterstate
        # take argmax
        # return action
        pass
