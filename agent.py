from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy
from model import Network

class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'

    def choose(self):
        return random.choice((0, 1, 2, 3))

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.learning_rate = -0.01 # negative to perform stochastic gradient ASCENT!

        self.policy_function = Network(
            dims=(16,8,4), \
            activation_funcs = [(sigmoid, sigmoid_prime),(sigmoid, sigmoid_prime)], \
            loss=(mse_loss, mse_loss_prime), \
            cost=cost, \
            seed=1
        )
        self.performance_function = 0 # this is just the score of an episode

    def update(self, state_history, action_history, reward_history):
        
        for t in range(len(state_history)):
            current_state = self.state_history[t]
            current_reward = self.reward_history[t]
            current_action = self.action_history[t]

            policy_eval = self.policy_function._forward(current_state)

            self.policy_function._backward(
                current_state, \
                current_action, \
                self.learning_rate*current_reward*(1/policy_eval[current_action])
            ) # TODO learning rate has other things multiplied to it

    def choose(self, state):
        return self.policy_function.inference(state)


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
experiment(agent=DumbAgent(), num_trials=100, dynamic_viz=True)


