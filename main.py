import yaml
import logging
import datetime

import torch
from torchinfo import summary

from nets.agent import Agent
from game.gameof2048 import Gameof2048

def main():
    experiment_start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
    NUM_ITERS = config['num_iters']
    NUM_GAMES_PER_ITER = config['num_games_per_iter']

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'nets/logs/experiment-{experiment_start_time}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting experiment {experiment_start_time} on device: {DEVICE}")
    
    agent = Agent(games_per_iter=NUM_GAMES_PER_ITER, device=DEVICE)

    for iter_idx in range(NUM_ITERS):
        games = [Gameof2048() for _ in range(NUM_GAMES_PER_ITER)]

        all_games_over = False
        while not all_games_over:
            # make state
            actions = agent.choose([game.board for game in games if not game.game_over])

            # make move
            all_moves_over_check = True
            for game_idx, game, action in enumerate(zip(games, actions)):
                if not game.game_over:
                    reward = game.do_move(direction=action)
                    all_moves_over_check = False
            
                    # TODO add reward to buffer
                else:
                    # TODO finish trajectory
                    # TODO death reward
                    pass

            all_games_over = all_moves_over_check

        average_score = sum([game.score for game in games])/NUM_GAMES_PER_ITER
        logger.info(f"Average score for Iter {iter_idx}: {average_score:.1f}")

        agent.update()

if __name__ == "__main__":
    main()
