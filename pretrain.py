import yaml
import logging
from datetime import datetime
from time import time
import sys
from statistics import stdev
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from nets.net import PolicyValueNet
from nets.dataset import PretrainingSet
from nets.agent import Agent
from eval import batch_eval

def pretrain():
    experiment_start_time = datetime.now()
    experiment_start_time_str = experiment_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
    EPOCHS = config['pretraining_epochs']
    BATCH_SIZE = config['pretraining_batch_size']

    NUM_GAMES_PER_ITER = config['num_games_per_iter']
    PLY = config['ply']

    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        filename=f'nets/logs/pretraining/pretraining-{experiment_start_time_str}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting experiment: {experiment_start_time_str} on device: {DEVICE}")
    logger.info(f"Pretraining for {EPOCHS} epochs at batch size: {BATCH_SIZE}")

    agent = Agent(
        ply=PLY, 
        games_per_iter=NUM_GAMES_PER_ITER, 
        device=DEVICE
    )

    dataset = PretrainingSet(data_path=f'./nets/data/data.csv')

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[0.9, 0.1])

    train_pretrain_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_pretrain_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    LEARNING_RATE = 5e-5
    LABEL_SMOOTHING = 0.05
    logger.info(f'Learning rate: {LEARNING_RATE} Label Smoothing: {LABEL_SMOOTHING}')

    # TODO proper learning reate schedule
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=LEARNING_RATE)
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for epoch in range(EPOCHS):
        ## training
        train_loss = 0
        prog_bar = tqdm(enumerate(train_pretrain_dataloader), total=(len(train_dataset)//BATCH_SIZE)+1)
        for batch_idx, (boards, moves) in prog_bar:
            optimizer.zero_grad()

            boards = boards.to(DEVICE)
            moves = moves.to(DEVICE)

            predictions, _ = agent.net(boards)

            loss = loss_fn(predictions, moves)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            batch_info_str = f'Train epoch {epoch} batch {batch_idx} completed with train loss: {loss.item():.5f}'
            logger.info(batch_info_str)
            prog_bar.set_description(batch_info_str)

        train_loss = train_loss/len(train_pretrain_dataloader)
        logger.info(f'> Epoch {epoch} completed with training loss: {train_loss}')

        ## validation
        with torch.no_grad():
            valid_loss = 0
            prog_bar = tqdm(enumerate(valid_pretrain_dataloader), total=(len(valid_dataset)//BATCH_SIZE)+1)
            for batch_idx, (boards, moves) in prog_bar:
                boards = boards.to(DEVICE)
                moves = moves.to(DEVICE)

                predictions, _ = agent.net(boards)

                loss = loss_fn(predictions, moves)

                valid_loss += loss.item()
                batch_info_str = f'Valid epoch {epoch} batch {batch_idx} completed with valid loss: {loss.item():.5f}'
                logger.info(batch_info_str)

            valid_loss = valid_loss/len(valid_pretrain_dataloader)
            logger.info(f'> Epoch {epoch} completed with valid loss: {train_loss}')

        ## evaluation
        scores = batch_eval(agent=agent, batch_size=BATCH_SIZE, doing_rl_training=False)
        max_score = max(scores)
        average_score = sum(scores)/len(scores)
        stdevs_score = stdev(scores)

        status_str = f"Epoch {epoch} Avg score: {average_score:.1f} Max score: {max_score} Stdev: {stdevs_score:.4f}"
        print(status_str)
        logger.info(status_str)
        prog_bar.set_description(status_str)

if __name__ == '__main__':
    pretrain()

