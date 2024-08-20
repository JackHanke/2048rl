from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import matplotlib.pyplot as plt
import numpy

class DumbAgent:
    def __init__(self):
        self.name = 'Dumb Agent'

    def choose(self):
        return random.choice((0, 1, 2, 3))

class QLearningAgent:
    def __init__(self):
        self.name = 'Q-Learning Agent'

    def update(): #
        pass

    def choose(): #
        pass

class ActorCriticAgent:
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


