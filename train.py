import yaml
import logging
import datetime

import torch
from torchinfo import summary

from nets.net import PolicyValueNet
from nets.agent import Agent
from game.gameof2048 import Gameof2048

def train():
    experiment_start_time = datetime.now()

    with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
    NUM_ITERS = config['num_iters']
    NUM_GAMES_PER_ITER = config['num_games_per_iter']

    EMBEDDING_DIM = config['embedding_dim']
    NUM_LAYERS = config['num_layers']

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'logs/experiment-{experiment_start_time}.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    agent = Agent(device=DEVICE)

    logger.info(f"Starting experiment on: {experiment_start_time}")
    for iter_idx in range(NUM_ITERS):
        agent.optimizer.zero_grad()

        games = [Gameof2048 for _ in range(NUM_GAMES_PER_ITER)]
        games_playing = [i for i in range(NUM_GAMES_PER_ITER)]

        all_games_over = False
        while not all_games_over:
            # make state
            actions = agent.choose([game.board for game in games if not game.game_over])

            # make move
            all_moves_over_check = True
            for game, action in zip(games, actions):
                if not game.game_over:
                    game.do_move(direction=action)
                    all_moves_over_check = False
                    
            all_games_over = all_moves_over_check


if __name__ == "__main__":
    train()
