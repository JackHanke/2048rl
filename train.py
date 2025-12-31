import yaml
import logging
import datetime

import torch
from torchinfo import summary

from nets.net import PolicyValueNet

def train():
    experiment_start_time = datetime.now()

    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    LEARNING_RATE = 1e-4

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

    model = PolicyValueNet(
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    summary_str = summary(model, input_size=(BATCH_SIZE, 4, 4, 17))
    model_summary_str = '\n'+str(summary_str)
    logger.info(model_summary_str)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # best_loss = float('inf')

    logger.info(f"Starting experiment on: {experiment_start_time}")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # TODO everything

        # TODO add game parallelism

if __name__ == "__main__":
    train()
