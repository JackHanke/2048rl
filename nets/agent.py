import yaml
from numpy import np
from math import log
import logging

import torch
from torchinfo import summary

from nets.buffer import Buffer
from nets.net import PolicyValueNet

def state_to_tensor(state_array: np.array) -> torch.tensor:
    return_tensor = torch.zeros((1, 17))
    return_tensor[0][0] = 18 # class token first
    for i in range(4):
        for j in range(4):
            return_tensor[0][(4*i)+j+1] = state_array[i][j]
    return return_tensor

with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(
            self, 
            device,
            ply: int = 0, 
            mode: str = 'inference',
        ):
        self.learning_rate = 1e-4
        self.batch_size = config['batch_size']
        self.net = PolicyValueNet(
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.buffer = Buffer()
        self.ply = ply
        self.mode = mode

        summary_str = summary(self.net, input_size=(self.batch_size, 4, 4, 17))
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)

    def train(self):
        self.mode = 'training'

    def eval(self):
        self.mode = 'inference'

    def choose(self, boards) -> torch.tensor:
        state = []
        for board in boards:
            state.append(state_to_tensor(board.board))
        state_tensor = torch.stack(state, dim=0)

        if self.ply == 0:
            logits, val = self.net(state_tensor)

            # legal filter
            for board_idx, board in enumerate(boards):
                for i in range(4):
                    if i not in board.legal_moves:
                        logits[board_idx][i] = -float('inf')

            if self.mode == 'inference':
                # argmax
                probs = torch.argmax(logits, dim=1)
                return torch.multinomial(probs, num_samples=1)
            elif self.mode == 'training':
                # sample
                return torch.nn.functional.softmax(logits, dim=1)

        else:
            # TODO expectimax
            pass

    def update(self):

        # TODO 

        pass

