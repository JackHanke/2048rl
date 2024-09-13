import matplotlib.pyplot as plt
from time import time
from game.gameof2048 import Gameof2048
from agents.dumbagent import DumbAgent
from agents.greedy import GreedyAgent
from agents.TDZeroApproxAgent.tdagent import TDApproxAgent
from models.ntuplenet import nTupleNetwork
from functions.tuplefuncs import *

def online_experiment(agent, num_trials, report_every, dynamic_viz=False, save=False, watch=False):
    print(f'Running experiment with {agent.name}...')
    start = time()
    scores = []
    running_avg_list = []
    for trial_num in range(num_trials):
        try:
            game = Gameof2048(agent=agent, watch=watch)
            final_score = game.play()
        # except KeyboardInterrupt:
        except KeyboardInterrupt:
            if save: agent.save()
            return scores, running_avg_list

        scores.append(final_score)
        # running_avg = sum(scores)/len(scores)
        running_avg = sum(scores[(-5*report_every):])/len(scores[(-5*report_every):])
        running_avg_list.append(running_avg)
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
                print(f'Trial {trial_num} achieved {final_score}, running av = {running_avg:.1f}, {((time()-start)/60):.4} mins')
    if dynamic_viz: plt.show()
    print(f'Trained on {num_trials} in {((time()-start)/3600):.2}hrs')
    if save:
        agent.save()
    return scores, running_avg_list

if __name__ == '__main__':
    agent_repeats = 1
    num_trials = 100000
    avg_scores = [0 for _ in range(num_trials)]
    for _ in range(agent_repeats):
        for lr in [0.01]:
            for agent in [
                TDApproxAgent(
                    version_num='2'
                    lmbda=1, 
                    n_step=1, 
                    discounting_param=1, 
                    reward_scale=1, 
                    learning_rate=lr,
                    state_val_approx=nTupleNetwork(tuple_map_class=TupleMap0())
                )
                ]:
                scores, running_avg_list = online_experiment(
                    agent=agent,
                    num_trials=num_trials, 
                    report_every=100,
                    dynamic_viz=False,
                    save=True,
                    watch=False
                )
                for index, val in enumerate(scores):
                    avg_scores[index] += val/agent_repeats

    # plt.scatter([i+1 for i in range(num_trials)], avg_scores, color='green')
    plt.scatter([i+1 for i in range(len(scores))], scores, color='red')
    plt.scatter([i+1 for i in range(len(running_avg_list))], running_avg_list, color='orange')
    plt.title(f'2048 {agent.name} Training Performance')
    plt.xlabel(f'Games Played')
    plt.ylabel(f'Final Score')    
    plt.show()
        
