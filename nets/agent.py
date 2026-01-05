import yaml
from numpy import np
from math import log
import logging

import torch
from torchinfo import summary
from torch.utils.data import DataLoader

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
            games_per_iter: int,
            mode: str = 'training',
        ):
        self.learning_rate = 1e-4
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.net = PolicyValueNet(
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.games_per_iter = games_per_iter
        self.buffer = Buffer(games_per_iter=self.games_per_iter)
        self.ply = config['ply']
        self.mode = mode

        self.epsilon = 0.2

        summary_str = summary(self.net, input_size=(self.batch_size, 4, 4, 17))
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)

    def _boards_to_tensor(self, boards) -> torch.tensor:
        state = []
        for board in boards:
            state.append(state_to_tensor(board.board))
        state_tensor = torch.stack(state, dim=0)
        return state_tensor

    def _get_legal_logits(self, boards) -> torch.tensor:
        state_tensor = self._boards_to_tensor(boards=boards)

        logits, val = self.net(state_tensor)
        # legal moves filter
        for board_idx, board in enumerate(boards):
            for i in range(4):
                if i not in board.legal_moves:
                    logits[board_idx][i] = -float('inf')
        return logits
    
    def _process_logits(self, logits: torch.tensor) -> torch.tensor:
        if self.mode == 'inference':
            # argmax
            probs = torch.argmax(logits, dim=1)
            result = torch.multinomial(probs, num_samples=1)
        elif self.mode == 'training':
            # sample
            result = torch.nn.functional.softmax(logits, dim=1)
        return result

    def train(self):
        self.mode = 'training'

    def eval(self):
        # self.optimizer.zero_grad()
        self.mode = 'inference'

    def choose(self, boards) -> torch.tensor:
        if self.ply == 0:
            logits = self._get_legal_logits(boards=boards)
            result = self._process_logits(logits=logits)
            return result
        else:
            # TODO expectimax
            pass

    def update(self):
        # 

        buffer_dataloader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
        for batch_idx, batch in enumerate(buffer_dataloader):
            # TODO logits, value, actions, returns, state, advantage
            
            # TODO send tensors to device
            # SPO Loss from: https://arxiv.org/abs/2401.16025
            policy_loss = None
            value_loss_fn = torch.nn.MSELoss()

            value_loss = value_loss_fn(value, returns)

            loss = policy_loss + value_loss

            loss.backward()
            self.optimizer.step()

            logger.debug(f'Batch {batch_idx} completed with loss: {loss.item():.6f}')
            pass

        # TODO checkpoint model

        self.buffer.reset()

