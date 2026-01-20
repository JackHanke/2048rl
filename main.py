import yaml
import logging
from datetime import datetime
from time import time
import sys
from statistics import stdev
from tqdm import tqdm

import torch
from torchsummary import summary

from nets.agent import Agent
from game.gameof2048 import Gameof2048

def main():
    experiment_start_time = datetime.now()
    experiment_start_time_str = experiment_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
    NUM_ITERS = config['num_iters']
    NUM_GAMES_PER_ITER = config['num_games_per_iter']
    PLY = config['ply']

    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        filename=f'nets/logs/experiment-{experiment_start_time_str}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting experiment: {experiment_start_time_str} on device: {DEVICE}")
    
    agent = Agent(ply=PLY, games_per_iter=NUM_GAMES_PER_ITER, device=DEVICE)
    logger.info(f"Agent Initialized (with {PLY}-ply search)") 
    logger.info(f"Playing {NUM_GAMES_PER_ITER} games per iteration for {NUM_ITERS} iterations")

    prog_bar = tqdm(range(NUM_ITERS))
    for iter_idx in prog_bar:
        games = [Gameof2048(game_idx=game_idx) for game_idx in range(NUM_GAMES_PER_ITER)]
        moves_played = 0
        all_games_over = False
        while not all_games_over:
            # make state
            boards = [game.board for game in games if not game.game_over]
            if len(boards) == 0: 
                all_games_over = True
            else:
                actions, logits, values = agent.choose(boards=boards, return_logits=True)

                # make move
                schedule = [game for game in games if not game.game_over]
                for playing_game_idx, game in enumerate(schedule):
                    action = actions[playing_game_idx].item()
                    reward = game.do_move(action=action)
                    moves_played += 1
                    agent.add(
                        board=game.board,
                        action=action,
                        reward=reward,
                        logits=logits[playing_game_idx],
                        value=values[playing_game_idx], 
                        game_idx=game.game_idx,
                    )
                    # if game.do_move ended the game, finish trajectory
                    if game.game_over:
                        agent.buffer.finish_trajectory(game_idx=game.game_idx)

        ## 
        scores = [int(game.board.score) for game in games]
        max_score = max(scores)
        average_score = sum(scores)/len(scores)
        stdevs_score = stdev(scores)

        status_str = f"Iter {iter_idx} Avg score: {average_score:.1f} Max score: {max_score} Stdev: {stdevs_score:.4f}"
        logger.info(status_str)
        prog_bar.set_description(status_str)
        
        agent.update()
        

    final_str = f'{NUM_ITERS} iterations of {NUM_GAMES_PER_ITER} games each completed in {(datetime.now() - experiment_start_time)}.'
    print(final_str)
    logger.info(final_str)

if __name__ == "__main__":
    main()
