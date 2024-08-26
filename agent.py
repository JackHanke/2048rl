from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
from model import Network, sigmoid, sigmoid_prime, softmax, LinearSoftmax, relu, relu_prime

class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'
        self.online = False
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def choose(self, state):
        return random.choice((0, 1, 2, 3))

    def update(self):
        pass

class Evolutionary:
    def __init__(self):
        pass

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.online = False
        self.learning_rate = -0.00001 # negative to perform stochastic gradient ASCENT
        self.gamma = 0.4 # discounting factor for reward
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
            # activation_funcs = [(relu, relu_prime),(relu, relu_prime),(softmax, softmax)], \
            # activation_funcs = [(sigmoid, sigmoid_prime),(sigmoid, sigmoid_prime),(softmax, softmax)], \
            # activation_funcs = [
            #     (sigmoid, sigmoid_prime), \
            #     (sigmoid, sigmoid_prime), \
            #     (sigmoid, sigmoid_prime), \
            #     (softmax, softmax) \
            # ], \
        self.policy_function = Network(
            dims=(16,32,16,4), \
            activation_funcs = [
                (relu, relu_prime), \
                (relu, relu_prime), \
                (softmax, softmax) \
            ], \
            seed=1
        )
        # self.policy_function = LinearSoftmax()
        def baseline(state): return 0
        self.baseline_function = baseline

    def update(self):
        return_val = 0

        max_weights = []
        max_grads = []

        for t in range(len(self.state_history)):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t] # TODO one hot?
            return_val = sum([(self.gamma**(k-t-1)) * (self.reward_history[k] - self.baseline_function(current_state)) for k in range(t+1,len(self.state_history))])

            policy_eval = self.policy_function._forward(current_state)
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            dude = max(policy_eval[current_action][0],0.01)
            learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)/dude
            # learning_rate_term = self.learning_rate*return_val*((self.gamma)**t)
            # print(learning_rate_term)
            # input()

            max_weight, max_grad = self.policy_function._backward(
                current_state, \
                current_action, \
                learning_rate_term
            )

            max_weights.append(max_weight)
            max_grads.append(max_grad)

        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        return max(max_weights), max(max_grads)

    def choose(self, state, invalid_moves):
        def prob_argmax(vec): return int(np.random.choice([i for i in range(len(vec))],1,p=vec)[0])
        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        # invalid action masking
        for action, prob in enumerate(activation):
            if action in invalid_moves:
                activation[action] = 0
        normalize = sum(activation)
        # if normalize < 0.000000000001:
        #     return np.argmax(activation)
        for action, prob in enumerate(activation):
            print(activation)
            activation[action] = prob/normalize
        return_val = prob_argmax(activation)
        # print(return_val, end=', ')
        # input()
        return return_val


class ActorCriticAgent:
    def __init__(self):
        self.name = 'Actor Critic Agent'

        self.value_function = 0
        self.policy_function = 0
        self.performance_function = 0

    def update(self):
        pass

    def choose(self):
        pass

# mean ~ 1096
# stdev ~ 549
# median ~ 1048
def experiment(agent, num_trials, dynamic_viz=False):
    scores = []
    weights = []
    grads = []
    start = time()
    for trial_num in range(num_trials):
        # final_score, best_tile = main(args=['--auto', '--viz'], agent=agent)
        try:
            final_score, best_tile, max_weight, max_grad = main(args=['--auto'], agent=agent)
        except Exception as e:
            input(e)
        scores.append(final_score)
        running_avg = mean(scores)

        weights.append(max_weight)
        running_avg_weights = mean(weights)

        grads.append(max_grad)
        running_avg_grads = mean(grads)
        if dynamic_viz:

            # plt.subplot(1,2,1)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Running Average = {running_avg:.1f} Points')

            # plt.subplot(1,2,2)
            # plt.scatter(trial_num, best_tile, c='pink')
            # plt.title(f'2048 {agent.name} best tile achieved')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Tile Exponent')

            # plt.subplot(1,2,2)
            # plt.scatter(trial_num, max_weight, c='yellow')
            # plt.scatter(trial_num, running_avg_weights, c='green')
            # plt.title(f'Max Weights for {agent.name}')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Max Weight, Running Average = {running_avg:.1f} Points')

            # plt.subplot(3,1,3)
            # plt.scatter(trial_num, max_grad, c='blue')
            # plt.scatter(trial_num, running_avg_grads, c='purple')
            # plt.title(f'Max Grad for {agent.name}')
            # plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            # plt.ylabel(f'Max Grad Norm, Running Average = {running_avg:.1f} Points')

            plt.pause(0.00001)
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

# baseline(num_trials=1000, verbose=True)
# experiment(agent=DumbAgent(), num_trials=1000, dynamic_viz=True) # 10,000 trials completed in 42.8 s
experiment(agent=REINFORCEMonteCarloPolicyGradientAgent(), num_trials=1000, dynamic_viz=True)


