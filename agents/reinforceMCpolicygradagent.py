import numpy as np
from models.ann import ArtificialNeuralNetwork
from functions.repfuncs import one_hot_state_rep
from functions.activations import leaky_relu, leaky_relu_prime, softmax
from functions.rlfuncs import prob_argmax

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.type = 'offline'
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        self.state_representation_function = one_hot_state_rep
        self.gamma = 1 # discounting factor for reward
        # self.policy_function = LinearSoftmax()
        self.policy_learning_rate = -0.009 # negative to perform stochastic gradient ASCENT
        self.policy_function = ArtificialNeuralNetwork(
            dims=(256,100, 60, 4), \
            activation_funcs = [
                (leaky_relu, leaky_relu_prime), \
                (leaky_relu, leaky_relu_prime), \
                (softmax, softmax) \
            ], \
            seed=1
        )
        self.baseline_learning_rate = -0.001 # negative to perform stochastic gradient ASCENT
        self.baseline_function = ArtificialNeuralNetwork(
            dims=(256,80,1), \
            activation_funcs = [
                (leaky_relu, leaky_relu_prime), \
                (leaky_relu, leaky_relu_prime)
            ], \
            seed=1
        )

    def update(self):
        return_val = 0
        for t in range(len(self.state_history)):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t]
            policy_eval = self.policy_function._forward(current_state)
            return_val = sum([(self.gamma**(k-t-1)) * (self.reward_history[k]) for k in range(t+1,len(self.state_history))])
            # predicted_state_val = self.baseline_function._forward(current_state)
            predicted_state_val = 0
            delta = (return_val - predicted_state_val)/1000
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            policy_lr_term = self.policy_learning_rate*delta*((self.gamma)**t)
            baseline_lr_term = self.policy_learning_rate*delta

            self.policy_function._backward(
                current_state, \
                current_action, \
                policy_lr_term
            )

            # self.baseline_function._backward(
            #     current_state, \
            #     current_action, \
            #     baseline_lr_term
            # )

        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def choose(self, state, invalid_moves):
        
        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        # invalid action masking
        for action, prob in enumerate(activation):
            if action in invalid_moves:
                activation[action] = 0
        normalize = sum(activation)
        
        for action, prob in enumerate(activation):
            activation[action] = prob/normalize
        return_val = prob_argmax(activation)
        return return_val
