from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
from model import Network, sigmoid, sigmoid_prime, softmax


class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'

    def choose(self):
        return random.choice((0, 1, 2, 3))

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.online = False
        self.learning_rate = -0.01 # negative to perform stochastic gradient ASCENT!
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

        self.policy_function = Network(
            dims=(16,8,4), \
            activation_funcs = [(sigmoid, sigmoid_prime),(softmax, softmax)], \
            seed=1
        )
        self.performance_function = 0 # this is just the score of an episode

    def update(self, state_history, action_history, reward_history):
        
        for t in range(len(state_history)-1,-1,-1):
            current_state = self.state_history[t]
            current_action = self.action_history[t] # TODO one hot?
            current_reward = self.reward_history[t]

            policy_eval = self.policy_function._forward(current_state)

            self.policy_function._backward(
                current_state, \
                current_action, \
                self.learning_rate*current_reward*(1/policy_eval[current_action])
            )

        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def choose(self, state):
        def prob_argmax(vec):
            return np.random.choice([i for i in range(len(vec))],1,p=vec)

        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        return int(prob_argmax(activation))


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

def baseline(num_trials, verbose=False):
    # mean ~ 1096
    # stdev ~ 549
    # median ~ 1048
    scores = []
    start = time()
    for _ in range(num_trials):
        val, _ = main(args=['--auto','True'], agent=DumbAgent())
        scores.append(val)
    print(f'completed {num_trials} in {(time()-start):.5}s')
    if verbose:
        print(f'mean(scores) for {num_trials} trials is {mean(scores)}')
        print(f'stdev(scores) for {num_trials} trials is {stdev(scores)}')
        print(f'median(scores) for {num_trials} trials is {median(scores)}')

def experiment(agent, num_trials, dynamic_viz=False):
    scores = []
    start = time()
    for trial_num in range(num_trials):
        final_score, best_tile = main(args=['--auto','True'], agent=agent)
        scores.append(final_score)
        running_avg = mean(scores)
        if dynamic_viz:
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score')
            plt.pause(0.00001)
    if dynamic_viz: plt.show()

# baseline(num_trials=1000, verbose=True)
# experiment(agent=DumbAgent(), num_trials=100, dynamic_viz=True)
experiment(agent=REINFORCEMonteCarloPolicyGradientAgent(), num_trials=100, dynamic_viz=True)


