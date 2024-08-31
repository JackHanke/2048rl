import matplotlib.pyplot as plt
from time import time
from macht.macht.term import main
from statistics import mean

import numpy as np
from agents.reinforceMCpolicygradagent import REINFORCEMonteCarloPolicyGradientAgent


def offline_experiment(agent, num_trials, dynamic_viz=False):
    scores = []
    weights = []
    grads = []
    start = time()
    for trial_num in range(num_trials):
        # final_score, best_tile = main(args=['--auto', '--viz'], agent=agent)
        try:
            final_score, best_tile = main(args=['--auto'], agent=agent)
        except Exception as e:
            input(e)
        scores.append(final_score)
        running_avg = mean(scores)

        if dynamic_viz:
            # plt.subplot(1,2,1)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Running Average = {running_avg:.1f} Points')

            plt.pause(0.00001)
        else:
            # print(f'Running average = {running_avg:.1f} \r\033[K', end='')
            if trial_num % 15 == 0: print(f'Running average after {trial_num} trials = {running_avg:.1f}')
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

if __name__ == '__main__':
    offline_experiment(
        agent=REINFORCEMonteCarloPolicyGradientAgent(), \
        num_trials=10000, \
        dynamic_viz=False
    )

