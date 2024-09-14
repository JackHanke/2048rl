import matplotlib.pyplot as plt
from time import time
from game.gameof2048 import Gameof2048
from agents.RandomAgent.randomagent import RandomAgent
from agents.TDZeroApproxAgent.tdagent import TDApproxAgent
from agents.greedy import GreedyAgent
from math import log
from models.ntuplenet import nTupleNetwork
from functions.tuplefuncs import *
import json

def benchmark(agent, num_games, report_every, dynamic_viz=False, save=False, watch=False):
    print(f'Benchmarking {agent.name}...')
    start = time()
    scores = []
    best_tile_array = [0 for _ in range(18)]
    best_score = 0
    for trial_num in range(num_games):
        try:
            game = Gameof2048(agent=agent, watch=watch)
            final_score = game.play()
            if final_score > best_score:
                best_score = final_score
                best_gameplay = game.gameplay
            best_tile_array[int(log(game.board.highest_tile, 2))] += (1/num_games)
        except KeyboardInterrupt:
            return scores

        scores.append(final_score)
        # running_avg = sum(scores)/len(scores)
        # running_avg = sum(scores[(-5*report_every):])/len(scores[(-5*report_every):])
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
    with open(f'agents/{agent.name}/gameplay-{agent.version_num}-1.json', 'w') as fout:
            json.dump(best_gameplay, fout)
    print(f'Benchmarked on {num_games} in {((time()-start)/3600):.2}hrs')
    print(f'Average Performance = {sum(scores)/len(scores)}')
    print(f'Best score achieved: {best_score}')
    print(f'Best tiles prob = {best_tile_array}')
    return scores

if __name__ == '__main__':
    agent_repeats = 1
    num_games = 1
    avg_scores = [0 for _ in range(num_games)]
    for _ in range(agent_repeats):
        # load_loc=f'agents/TDZeroApproxAgent/TDZeroApproxAgent-model-2.json'
        # agent = TDApproxAgent(
        #     version_num='2',
        #     lmbda=0, 
        #     n_step=1, 
        #     discounting_param=1, 
        #     reward_scale=1, 
        #     learning_rate=0.01,
        #     state_val_approx=nTupleNetwork(tuple_map_class=TupleMap0(), load_loc=load_loc),
        # )
        agent = RandomAgent()
        scores = benchmark(
            agent=agent,
            num_games=num_games, 
            report_every=25,
            dynamic_viz=False,
            watch=False
        )
        for index, val in enumerate(scores):
            avg_scores[index] += val/agent_repeats

    # plt.scatter([i+1 for i in range(num_games)], avg_scores, color='green')
    # plt.show()
