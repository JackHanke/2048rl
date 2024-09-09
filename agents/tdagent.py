from models.ann import ArtificialNeuralNetwork
from models.ntuplenet import nTupleNetwork
from functions.activations import *
from models.linear import Linear
from functions.repfuncs import identity_rep, simple_exponent_state_rep
from functions.rlfuncs import epsilon_greedy, better_argmax_dict
from statistics import mean
import numpy as np
import json

class TDApproxAgent:
    def __init__(self):
        self.name = 'TD(0) Approx Agent'
        self.type = 'online'
        self.lmbda = 0
        self.n_step = 1
        self.discounting_param = 1
        self.learning_rate = 0.0025
        self.reward_scale = 1
        self.staterepfunc = identity_rep
        self.state_value_function_approx = nTupleNetwork()
        self.temp_val = 0 # stores r + V(s') to avoid extra eval

    def save(self, loc):
        with open(loc, 'w') as fout:
            json.dump(your_list_of_dict, fout)

    def load(self, loc):
        with open(loc, 'r') as fin:
            json.load(your_list_of_dict, fin)

    def choose(self, state, afterstates):
        state_rep = self.staterepfunc(state)
        predicted_rewards_for_each_action = {}
        # valid_moves is a dict with legal actions as keys as a list of tuples of (reward, future states (with tiles spawned in) and their respective probs)
        for tup in afterstates:
            pred_val = tup[1] + self.state_value_function_approx.forward(self.staterepfunc(tup[2]))
            # print(tup)
            # print(f'agent preds (r = {tup[1]}) + (V(af s) = {pred_val-tup[1]}) = {pred_val}')
            predicted_rewards_for_each_action[tup[0]] = pred_val
        try:
            chosen_action = better_argmax_dict(predicted_rewards_for_each_action)
            self.temp_val = predicted_rewards_for_each_action[chosen_action]
        except Exception as e:
            print('errored at this state:')
            print(e)
            print(predicted_rewards_for_each_action)
            print(state)
        return chosen_action

    def update(self, afterstate, state, chosen_action, afterstates):
        state_rep = self.staterepfunc(state)
        val_old_afterstate = self.state_value_function_approx.forward(afterstate) # TODO fix this re-eval
        self.state_value_function_approx.backward(
            activation = afterstate,
            label = (self.temp_val - val_old_afterstate),
            learning_rate=self.learning_rate
        )