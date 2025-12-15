# benchmarking game and agent
import json
from time import time
from math import log
import matplotlib.pyplot as plt

from game.gameof2048 import Gameof2048
from scratch.agents.RandomAgent.randomagent import RandomAgent

# benchmark game implementation 
def benchmark_game(
        num_moves: int
    ):

    agent = RandomAgent()

    start_time = time()
    moves_played = 0
    games_played = 0
    while moves_played < num_moves:
        game = Gameof2048(agent=agent, watch=False)

        final_score = game.play()
        games_played += 1

        moves_played += game.moves

    total_time = time() - start_time

    moves_per_second = moves_played/total_time
    # TODO variance calculation

    print(f'Played {moves_played:,} moves over {games_played} games in {total_time:.2f} seconds.')
    print(f'Moves/sec: {moves_per_second:.1f} Sec/Moves: {(1/moves_per_second):.6f}')
        

# benchmark agent for performance for num_games, reporting score every report_every
def benchmark_agent(
        agent,
        num_games: int, 
        report_every: int, 
        dynamic_viz: bool = False, 
        save: bool = False, 
        watch: bool = False
    ):

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
    return scores, best_tile_array

if __name__ == '__main__':
    
    # num_games = 1000
    # avg_scores = [0 for _ in range(num_games)]
    # agent = RandomAgent()
    # scores = benchmark_agent(
    #     agent=agent,
    #     num_games=num_games, 
    #     report_every=25,
    #     dynamic_viz=False,
    #     watch=False
    # )

    num_moves = 20_000
    benchmark_game(num_moves=num_moves)
