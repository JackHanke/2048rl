from models.ann import ArtificialNeuralNetwork
from functions.activations import *
from models.linear import Linear
from functions.environmentfuncs import one_hot_state_rep
from functions.rlfuncs import epsilon_greedy, better_argmax_dict
from statistics import mean
import numpy as np
# read this: https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html

# on policy 
class MonteCarloApproxAgent:
    def __init__(self):
        self.name = 'Monte Carlo Approx Agent'
        self.type = 'offline'
        self.lmbda = 0
        self.n_step = 1
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.discounting_param = 1
        self.epsilon = 0.05
        self.learning_rate = -0.00001
        self.state_representation_function = one_hot_state_rep
        # self.state_value_function_approx = Linear(dims=(256,1), seed=1)
        self.state_value_function_approx = ArtificialNeuralNetwork(
            dims=(256, 100, 1), \
            activation_funcs = [
                (leaky_relu, leaky_relu_prime), \
                (relu, relu_prime)
            ], \
            seed=1
        )

    def update(self):
        return_val = 0
        for t in range(len(self.state_history)):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t]
            predicted_state_val = self.state_value_function_approx._forward(current_state)
            return_val = sum([(self.discounting_param**(k-t-1)) * (self.reward_history[k]) for k in range(t+1,len(self.state_history))])
            delta = (return_val - predicted_state_val)
            lr_term = self.learning_rate*delta
            # print(f'lr_term = {lr_term}')
            self.state_value_function_approx._backward(
                activation=current_state, \
                label=None, \
                learning_rate=lr_term
            )
        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def choose(self, state, valid_moves):
        predicted_state_vals = 0
        predicted_rewards_for_each_action = {}
        # valid_moves is a dict with legal actions as keys as a list of tuples of (reward, future states (with tiles spawned in) and their respective probs)
        for key, value in valid_moves.items():
            val = 0
            for thing in value:
                future_state_rep = self.state_representation_function(thing[1])
                future_state_rep = np.array([future_state_rep]).transpose()
                future_state_approx_val = self.state_value_function_approx._forward(future_state_rep)
                val += thing[0] + (future_state_approx_val*thing[2] )
            predicted_rewards_for_each_action[key] = (val/len(value)) # average value of the action
        chosen_action = better_argmax_dict(predicted_rewards_for_each_action)
        return chosen_action
