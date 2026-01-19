import yaml
import logging
from datetime import datetime
from time import time

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

    for iter_idx in range(NUM_ITERS):
        games = [Gameof2048() for _ in range(NUM_GAMES_PER_ITER)]

        all_games_over = False
        while not all_games_over:
            # make state
            boards = [game.board for game in games if not game.game_over]
            actions, logits, values = agent.choose(boards=boards, return_logits=True)

            # make move
            all_moves_over_check = True
            playing_game_idx = 0
            for game_idx, game in enumerate(games):
                if not game.game_over:
                    all_moves_over_check = False
                    action = actions[playing_game_idx].item()
                    reward = game.do_move(action=action)
                    agent.add(
                        board=game.board,
                        action=action,
                        reward=reward,
                        logits=logits[playing_game_idx],
                        value=values[playing_game_idx], 
                        game_idx=game_idx,
                    )
                    playing_game_idx += 1
                elif not agent.buffer.finished_buffer[game_idx]:
                    agent.buffer.finish_trajectory(game_idx=game_idx)

            all_games_over = all_moves_over_check

        average_score = sum([game.board.score for game in games])/NUM_GAMES_PER_ITER
        logger.info(f"Average score for Iter {iter_idx}: {average_score:.1f}")

        agent.update()

    final_str = f'{NUM_ITERS} iterations of {NUM_GAMES_PER_ITER} games each completed in {(datetime.now() - experiment_start_time)}.'
    print(final_str)
    logger.info(final_str)

if __name__ == "__main__":
    main()
