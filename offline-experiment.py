import matplotlib.pyplot as plt
from time import time
from statistics import mean
from math import sqrt
import numpy as np
from macht.macht.term import main
from agents.reinforceMCpolicygradagent import REINFORCEMonteCarloPolicyGradientAgent
from agents.tdagent import MonteCarloApproxAgent

def offline_experiment(agent, num_trials, report_every=15, dynamic_viz=False):
    scores = []
    mses = []
    start = time()
    for trial_num in range(num_trials):
        # game_time = time()
        try:
            final_score, best_tile, mse = main(args=['--auto'], agent=agent)
        except Exception as e:
            input(e)
        # print(f'full process time = {(time()-game_time):.8f}')
        # input()
        scores.append(final_score)
        mses.append(mse)
        running_avg = mean(scores[(-1*report_every):])
        running_mse_avg = mean(mses[(-1*report_every):])
        if dynamic_viz and trial_num % report_every == 0:
            plt.subplot(2, 2, 1)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Last {report_every} Running Average = {running_avg:.1f} Points')

            plt.subplot(2, 2, 2)
            plt.scatter(trial_num, mse, c='red')
            plt.scatter(trial_num, running_mse_avg, c='orange')
            plt.title(f'2048 {agent.name} MSE')
            plt.xlabel(f'Trial Number (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'MSE, Last {report_every} Running Average = {running_mse_avg:.1f} Points')
            plt.pause(0.00001)
        else:
            if trial_num % report_every == 0: 
                print(f'Running avg after {trial_num} games = {running_avg:.1f}, MSE = {sqrt(mse)}')
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

if __name__ == '__main__':
    # offline_experiment(
    #     agent=REINFORCEMonteCarloPolicyGradientAgent(), \
    #     num_trials=10000, \
    #     dynamic_viz=False
    # )
    offline_experiment(
        agent=MonteCarloApproxAgent(),
        num_trials=50000, 
        report_every=50,
        dynamic_viz=False
    )
