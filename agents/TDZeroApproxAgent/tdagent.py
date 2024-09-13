import numpy as np
import json
import random
from models.ntuplenet import nTupleNetwork
from models.ann import ArtificialNeuralNetwork
from functions.activations import *
from models.linear import Linear
from functions.repfuncs import identity_rep, simple_exponent_state_rep
from functions.rlfuncs import epsilon_greedy, better_argmax_dict
from functions.tuplefuncs import *

class TDApproxAgent:
    def __init__(self, lmbda, n_step, discounting_param, reward_scale, learning_rate, load_loc=None):
        self.name = 'TDZeroApproxAgent'
        self.type = 'online'
        self.lmbda = lmbda
        self.n_step = n_step
        self.discounting_param = discounting_param
        self.reward_scale = reward_scale
        self.learning_rate = learning_rate
        self.staterepfunc = identity_rep
        self.load_loc = load_loc
        self.state_value_function_approx = nTupleNetwork(tuple_map_class=TupleMap3(), load_loc=self.load_loc)
        self.temp_val = 0 # stores r + V(s') to avoid extra eval
        self.delta_history = []
        self.reward_history = []

    def save(self, loc):
        # TODO add movel versioning, use path package
        
        # model versioning
        # model_ver = 0
        # path_str = f'agents/{agent.name}/{agent.name}-model'
        # while True:
        #     try:
        #         with open(path_str+'-'+str(model_ver)+'.json', 'r') as fout: 
        #             pass
        #         model_ver += 1
        #     except FileNotFoundError:
        #         agent.save(loc=path_str+'-'+str(model_ver))
        #         break
        
        with open(loc+'.json', 'w') as fout:
            json.dump(self.state_value_function_approx.lookup_array, fout)

        with open(loc+'-params.json', 'w') as fout:
            params_dict = {
                "lmbda": self.lmbda,
                "n_step": self.n_step,
                "discounting_param": self.discounting_param,
                "reward_scale": self.reward_scale,
                "learning_rate": self.learning_rate
            }
            json.dump(params_dict, fout)
        
    def choose(self, state, afterstates):
        state_rep = self.staterepfunc(state)
        predicted_rewards_for_each_action = {}
        # afterstates contains a lists of tuples of (action, reward, afterstate)
        for tup in afterstates:
            pred_reward = tup[1]/self.reward_scale
            pred_stateval = self.state_value_function_approx.forward(self.staterepfunc(tup[2]))
            pred_val = pred_reward + self.discounting_param*pred_stateval
            predicted_rewards_for_each_action[tup[0]] = pred_val

        chosen_action = better_argmax_dict(predicted_rewards_for_each_action)
        self.temp_val = predicted_rewards_for_each_action[chosen_action]
        return chosen_action

    def update(self, afterstate, state, chosen_action, afterstates):
        if self.load_loc is not None: return # skip updates for loaded in model
        state_rep = self.staterepfunc(state)
        val_old_afterstate = self.state_value_function_approx.forward(afterstate) # TODO fix this re-eval
        self.state_value_function_approx.backward(
            activation = afterstate,
            label = (self.temp_val - val_old_afterstate),
            learning_rate=self.learning_rate
        )