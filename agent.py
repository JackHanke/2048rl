from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
from model import Network, sigmoid, sigmoid_prime, softmax, LinearSoftmax

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

class REINFORCEMonteCarloPolicyGradientAgent:
    def __init__(self):
        self.name = 'REINFORCE Monte Carlo Policy Gradient Agent'
        self.online = False
        self.learning_rate = -0.01 # negative to perform stochastic gradient ASCENT!
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]
        # self.policy_function = Network(
        #     dims=(16,8,4), \
        #     activation_funcs = [(sigmoid, sigmoid_prime),(softmax, softmax)], \
        #     seed=1
        # )

        self.policy_function = LinearSoftmax()

    def update(self):
        return_val = 0
        for t in range(len(self.state_history)-1,-1,-1):
            current_state = self.state_history[t]
            current_state = np.array([current_state]).transpose()
            current_action = self.action_history[t] # TODO one hot?
            return_val += self.reward_history[t]

            policy_eval = self.policy_function._forward(current_state)
            # print(policy_eval)
            # print(self.learning_rate*return_val*(1/policy_eval[current_action][0]))
            # input()
            self.policy_function._backward(
                current_state, \
                current_action, \
                self.learning_rate*return_val*(1/policy_eval[current_action][0])
            )

        # flush history
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        print('update completed')

    def choose(self, state):
        def prob_argmax(vec):
            return int(np.random.choice([i for i in range(len(vec))],1,p=vec)[0])

        state = np.array([state]).transpose()
        activation = self.policy_function._forward(state).transpose()[0]
        # print(activation)
        
        return prob_argmax(activation)


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
    start = time()
    for trial_num in range(num_trials):
        final_score, best_tile = main(args=['--auto', '--viz'], agent=agent)
        # final_score, best_tile = main(args=['--auto'], agent=agent)
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
    print(f'Completed {num_trials} in {(time()-start):.5}s')

# baseline(num_trials=1000, verbose=True)
# experiment(agent=DumbAgent(), num_trials=10, dynamic_viz=False) # 10,000 trials completed in 42.8 s
experiment(agent=REINFORCEMonteCarloPolicyGradientAgent(), num_trials=100, dynamic_viz=True)


