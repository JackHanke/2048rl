from models.ann import ArtificialNeuralNetwork
from models.ntuplenet import nTupleNetwork
from functions.activations import *
from models.linear import Linear
from functions.repfuncs import one_hot_state_rep, simple_exponent_state_rep
from functions.rlfuncs import epsilon_greedy, better_argmax_dict
from statistics import mean
import numpy as np

class TDApproxAgent:
    def __init__(self):
        self.name = 'Monte Carlo Approx Agent'
        self.type = 'online'
        self.lmbda = 0
        self.n_step = 1
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.discounting_param = 1
        self.learning_rate = 0.0025
        self.state_representation_function = simple_exponent_state_rep
        self.state_value_function_approx = nTupleNetwork(num_tuples=17)


    def choose(self, state, afterstates):
        state_rep = self.state_representation_function(state)
        predicted_state_vals = 0
        predicted_rewards_for_each_action = {}
        # valid_moves is a dict with legal actions as keys as a list of tuples of (reward, future states (with tiles spawned in) and their respective probs)
        for tup in afterstates:
            predicted_rewards_for_each_action[tup[0]] = tup[1] + \
            self.state_value_function_approx.forward(self.state_representation_function(tup[2]))
        chosen_action = better_argmax_dict(predicted_rewards_for_each_action)
        return chosen_action


    def update(self, afterstate, state, chosen_action, afterstates):
        state_rep = self.state_representation_function(state)

        afterstates_info = afterstates[chosen_action]
        reward, future_afterstate = afterstates_info[1], afterstates_info[2]

        # old_afterstate_rep = self.state_representation_function(afterstate)
        val_old_afterstate = self.state_value_function_approx.forward(afterstate)
        val_future_afterstate = self.state_value_function_approx.forward(future_afterstate)

        self.state_value_function_approx.backward(
            activation = afterstate,
            label = (reward + val_future_afterstate - val_old_afterstate),
            learning_rate=self.learning_rate
        )