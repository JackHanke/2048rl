import yaml
import numpy as np
from math import log
import logging

import torch
from torchsummary import summary
from torch.utils.data import DataLoader

from nets.buffer import Buffer
from nets.net import PolicyValueNet

with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(
            self, 
            ply: int,
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
        self.ply = ply
        self.mode = mode
        self.device=device

        self.epsilon = 0.2

        summary_str = summary(self.net, input_size=(self.batch_size, 4, 4, 17))
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)

    def _loss_fn(self, 
            predictions: torch.tensor,
            observations: torch.tensor, 
            actions: torch.tensor, 
            logits: torch.tensor,
            advantages: torch.tensor,
            returns: torch.tensor,
            values: torch.tensor,
        ):
        # loss from SPO paper: https://arxiv.org/abs/2401.16025
        ratio = torch.exp(torch.nn.functional.log_softmax(predictions) - logits)
        policy_loss = torch.mul(ratio, advantages) + \
            torch.mul(
                (-1/(2*self.epsilon))*torch.abs(advantages), 
                torch.square(ratio-1)
            )

        value_loss_fn = torch.nn.MSELoss()
        value_loss = value_loss_fn(values, returns)

        loss = policy_loss + value_loss
        return loss
    
    def _preprocess_reward(self, reward: float):
        if reward <= 0: return 0
        return log(reward, 2)

    def _get_legal_logits(self, boards) -> torch.tensor:
        state_tensor = self._boards_to_tensor(boards=boards)

        logits, values = self.net(state_tensor)
        # legal moves filter
        for board_idx, board in enumerate(boards):
            for i in range(4):
                if i not in board.legal_moves:
                    logits[board_idx][i] = -float('inf')
        return logits, values
    
    def _process_logits(self, logits: torch.tensor) -> torch.tensor:
        if self.mode == 'inference':
            # argmax
            result = torch.argmax(logits, dim=1) 
        elif self.mode == 'training':
            # sample
            probs = torch.nn.functional.softmax(logits, dim=1)
            result = torch.multinomial(probs, num_samples=1)
        return result

    def _boards_to_tensor(self, boards) -> torch.tensor:
        '''takes a list of n boards and makes an n,17 tensor with tile "tokens"'''
        state_tensor = torch.zeros((len(boards), 17), dtype=torch.int)
        for board_idx, board in enumerate(boards):
            state_tensor[board_idx][0] = 17 # class token first
            for i in range(4):
                for j in range(4):
                    state_tensor[board_idx][(4*i)+j+1] = int(board.board[i][j])
        return state_tensor

    def train(self):
        self.mode = 'training'

    def eval(self):
        # self.optimizer.zero_grad()
        self.mode = 'inference'

    def choose(self, boards, return_logits:bool = False) -> torch.tensor:
        if self.ply == 0:
            logits, values = self._get_legal_logits(boards=boards)
            result = self._process_logits(logits=logits)
            if return_logits: return result, logits, values
            return result
        else:
            # TODO expecti-negamax
            pass

    def add(self,
            board, 
            action: torch.tensor, 
            reward: torch.tensor, 
            logits: torch.tensor,
            value: torch.tensor, 
            game_idx: int = 0
        ):

        board_tensor = self._boards_to_tensor(boards=[board]).to(self.device)
        processed_reward = self._preprocess_reward(reward=reward)

        self.buffer.add(
            observation=board_tensor,
            action=action,
            reward=processed_reward,
            logits=logits,
            value=value,
            game_idx=game_idx,
        )

    def update(self):
        # 
        self.buffer.flatten_buffers()

        buffer_dataloader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
        for batch_idx, (observations, actions, logits, advantages, returns) in enumerate(buffer_dataloader):
            # 
            predictions, values = self.net(observations)

            loss = self._loss_fn(predictions, observations, actions, logits, advantages, values, returns)

            loss.backward()
            self.optimizer.step()

            logger.debug(f'Batch {batch_idx} completed with loss: {loss.item():.6f}')

        self.buffer.reset()

