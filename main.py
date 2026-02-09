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
from nets .net import PolicyValueNet
from eval import batch_eval


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
        filename=f'nets/logs/finetuning/experiment-{experiment_start_time_str}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting experiment: {experiment_start_time_str} on device: {DEVICE}")
    
    net = PolicyValueNet(
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        device=DEVICE
    ).to(DEVICE)

    load_model = True
    if load_model:
        LOAD_MODEL_PATH = f'./nets/models/pretraining/2026-02-08-00:14:07_57_3239.09_0.693.pth'
        net.load_state_dict(torch.load(LOAD_MODEL_PATH, weights_only=True), strict=False)
        logger.info(f'Beginning from checkpoint: {LOAD_MODEL_PATH}')
    else:
        logger.info(f'Beginning from scratch.')

    agent = Agent(
        ply=PLY,
        net = net,
        games_per_iter=NUM_GAMES_PER_ITER, 
        device=DEVICE,
        mode='training'
    )

    logger.info(f"Agent Initialized (with {PLY}-ply search)") 
    logger.info(f"Playing {NUM_GAMES_PER_ITER} games per iteration for {NUM_ITERS} iterations")

    prog_bar = tqdm(range(NUM_ITERS))
    for iter_idx in prog_bar:
        scores = batch_eval(agent=agent, batch_size=NUM_GAMES_PER_ITER, doing_rl_training=True)
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
