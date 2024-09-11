import matplotlib.pyplot as plt
from time import time
from macht.macht.term import main
from game.gameof2048 import Gameof2048
from agents.dumbagent import DumbAgent
from agents.tdagent import TDApproxAgent
from agents.greedy import GreedyAgent

def online_experiment(agent, num_trials, report_every, dynamic_viz=False, save=False, watch=False):
    print(f'Running experiment with {agent.name}...')
    start = time()
    scores = []
    for trial_num in range(num_trials):
        try:
            game = Gameof2048(agent=agent, watch=watch)
            final_score = game.play()

            # for tup_index, tup in enumerate(game.gameplay):
            #     if (tup_index % 4) ==0:
            #         print('afterstate:')
            #         print(tup)
            #     if (tup_index % 4) ==1:
            #         print('state:')
            #         print(tup)
            #     if (tup_index % 4) ==2:
            #         print(f'action: {tup}', end=' ')
            #     if (tup_index % 4) ==3:
            #         print(f'reward: {tup}')
            # input()
        except KeyboardInterrupt:
            if save: agent.save(loc=f'models/{agent.name}-model.json')
            return -1

        scores.append(final_score)
        # running_avg = sum(scores)/len(scores)
        running_avg = sum(scores[(-5*report_every):])/len(scores[(-5*report_every):])
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
                print(f'Trial {trial_num} achieved {final_score}')
                # print(f'Running avg after {trial_num} games = {running_avg:.1f}')
    if dynamic_viz: plt.show()
    print(f'Completed {num_trials} in {(time()-start):.5}s')
    return scores

if __name__ == '__main__':
    agent_repeats = 1
    num_trials = 10000
    avg_scores = [0 for _ in range(num_trials)]
    for _ in range(agent_repeats):
        for lr in [0.1]:
            for agent in [
                TDApproxAgent(
                    lmbda=1, 
                    n_step=1, 
                    discounting_param=1, 
                    reward_scale=1, 
                    learning_rate=lr
                )
                ]:
                scores = online_experiment(
                    agent=agent,
                    num_trials=num_trials, 
                    report_every=100,
                    dynamic_viz=True,
                    save=False,
                    watch=False
                )
                for index, val in enumerate(scores):
                    avg_scores[index] += val/agent_repeats

    # plt.scatter([i+1 for i in range(num_trials)], avg_scores, color='green')
    # plt.show()
        


