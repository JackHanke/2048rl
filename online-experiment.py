import matplotlib.pyplot as plt
from time import time
from macht.macht.term import main
from statistics import mean
from game.gameof2048 import Gameof2048
from agents.dumbagent import DumbAgent
from agents.tdagent import TDApproxAgent
from agents.greedy import GreedyAgent

def online_experiment(agent, num_trials, report_every, dynamic_viz=False, save=False):
    print(f'Running experiment with {agent.name}...')
    scores = []
    start = time()
    for trial_num in range(num_trials):
        try:
            game = Gameof2048(agent=agent)
            final_score = game.play()
        except KeyboardInterrupt:
            if save: agent.save(loc=f'models/{agent.name}-model.json')
            return -1

        scores.append(final_score)
        running_avg = mean(scores)
        # running_avg = mean(scores[(-1*report_every):])
        if dynamic_viz and trial_num % report_every == 0:
            # plt.subplot(2, 2, 1)
            # plt.subplot(2, 2, 2)
            plt.scatter(trial_num, final_score, c='red')
            plt.scatter(trial_num, running_avg, c='orange')
            plt.title(f'2048 {agent.name} Score')
            plt.xlabel(f'Trial #{trial_num} (Completed in {(time()-start):.2f}s)')
            plt.ylabel(f'Final Score, Last {report_every} Running Average = {running_avg:.1f} Points')
            plt.pause(0.00001)
        else:
            if trial_num % report_every == 0: 
                print(f'Running avg after {trial_num} games = {running_avg:.1f}')
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')

if __name__ == '__main__':
    for agent in [TDApproxAgent()]:
        online_experiment(
            agent=agent,
            num_trials=50000, 
            report_every=100,
            dynamic_viz=False,
            save=False
        ) 


