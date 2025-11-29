## NOTE this is really bad practice but ive got to get this done

from time import time
import torch 
import numpy as np
import pickle
from math import log

import neat.experiments.gameof2048.config as c
from neat.phenotype.feed_forward import FeedForwardNet
from neat.experiments.gameof2048.gameof2048 import Gameof2048
from neat.visualize import draw_net

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
    
    print(f'Benchmarked on {num_games} in {((time()-start)/3600):.2}hrs')
    print(f'Average Performance = {sum(scores)/len(scores)}')
    print(f'Best score achieved: {best_score}')
    print(f'Best tiles prob = {best_tile_array}')
    return scores, best_tile_array


class NEATAgent:
    def __init__(self, phenotype, device):
        self.name = 'NEATAgent'
        self.type = 'offline'
        self.version_num = 0
        self.phenotype = phenotype
        self.device = device

    def choose(self, state, afterstates):
        state_arr = np.ndarray.flatten(state)

        one_hot_arr = np.zeros(11*16)
        for loc, tile in enumerate(state_arr):
            # first 16 are 1 if 0, else 0, next 16 is 1 if the 2 tile, else 0, next 16 for 4 tile, etc.
            tile_index = 0 if tile == 0 else int(log(tile, 2))
            one_hot_arr[16*tile_index + loc] = 1

        stateTensor = torch.unsqueeze(torch.Tensor(one_hot_arr).to(self.device),0)

        predictionTensor = self.phenotype(stateTensor)

        # only choose among legal moves
        for move in range(4):
            if move not in [tup[0] for tup in afterstates]:
                predictionTensor[0][move] = float('-inf')
        
        chosen_action = torch.argmax(predictionTensor)

        return chosen_action

    def update(self, **kwargs): return


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_genome_path = 'neat/experiments/gameof2048/checkpoints/exp_2025-11-21 00:06:46.551707_gen_170.pickle'
# with open(best_genome_path)
with open(best_genome_path, 'rb') as f:
    best_genome = pickle.load(f)

best_phenotype = FeedForwardNet(best_genome, c.Gameof2048Config)

agent = NEATAgent(phenotype=best_phenotype, device=DEVICE)

# num_games = 2

# scores, best_tile_array = benchmark(
#     agent=agent,
#     num_games=num_games, 
#     report_every=25,
#     dynamic_viz=False,
#     watch=False
# )

# print([int(score) for score in scores])
# print('-'*50)
# print(best_tile_array)

print(best_genome)
print('-'*50)
print(best_phenotype)

# draw_net(best_genome, view=True, filename='./images/2048-solution', show_disabled=True)
