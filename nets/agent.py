import yaml
import logging
import numpy as np
from time import time
from math import log

import torch
from torchinfo import summary
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from game.board import Board
from nets.net import PolicyValueNet
from nets.buffer import Buffer, TDBuffer

from statistics import stdev

with open('config.yaml', 'r') as file: config = yaml.safe_load(file)
logger = logging.getLogger(__name__)

class SPOAgent:
    def __init__(
            self, 
            ply: int,
            device,
            games_per_iter: int,
            net,
            mode: str = 'training',
        ):
        self.learning_rate = 1e-5
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.games_per_iter = games_per_iter
        self.buffer = Buffer(games_per_iter=self.games_per_iter)
        self.ply = ply
        self.mode = mode
        self.device=device

        ## loss hyperparameters
        self.epsilon = 0.2  # PPO/SPO epsilon
        self.c_1 = 0.1        # weight to value loss term
        # self.c_2 = 1e-5    # weight of policy entropy term


        summary_str = summary(self.net, input_size=(self.batch_size, 4, 4, 17))
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)
        logger.info(f'Agent trains for epochs {self.epochs} at batch size: {self.batch_size} ')

    def _loss_fn(self, 
            predictions: torch.tensor,
            observations: torch.tensor, 
            actions: torch.tensor, 
            logits: torch.tensor,
            advantages: torch.tensor,
            returns: torch.tensor,
            values: torch.tensor,
        ):
        ## policy loss
        # loss from SPO paper: https://arxiv.org/abs/2401.16025

        all_probs_ratio = torch.exp(torch.nn.functional.log_softmax(predictions, dim=1) - \
                                    torch.nn.functional.log_softmax(logits, dim=1))
        ratio = all_probs_ratio[torch.arange(actions.shape[0]), actions]

        policy_losses = torch.mul(ratio, advantages) + \
            torch.mul(
                (-1/(2*self.epsilon))*torch.abs(advantages), 
                torch.square(ratio-1)
            )
        policy_loss = torch.mean(policy_losses, dim=0)

        ## value loss 
        value_loss_fn = torch.nn.MSELoss()
        value_loss = value_loss_fn(values, returns)

        ## entropy loss
        # entropy_loss = torch.mean(Categorical(probs=torch.nn.functional.softmax(predictions, dim=1)).entropy(), dim=0)

        loss = policy_loss + self.c_1*value_loss # + self.c_2*entropy_loss
        return loss
    
    def _preprocess_reward(self, reward: float):
        if reward <= 0: return 0
        return log(reward, 2)

    def _boards_to_tensor(self, boards) -> torch.tensor:
        '''takes a list of n boards and makes an n,17 tensor with tile "tokens"'''
        state_tensor = torch.zeros((len(boards), 17), dtype=torch.int).to(self.device)
        for board_idx, board in enumerate(boards):
            state_tensor[board_idx][0] = 17 # class token first
            for i in range(4):
                for j in range(4):
                    state_tensor[board_idx][(4*i)+j+1] = int(board.board[i][j])
        return state_tensor

    def _get_legal_logits(self, boards) -> torch.tensor:
        state_tensor = self._boards_to_tensor(boards=boards)

        logits, values = self.net(state_tensor)
        # logits = self.net(state_tensor)
        # values = None

        legal_logits = logits.detach().clone()
        # legal moves filter
        for board_idx, board in enumerate(boards):
            for i in range(4):
                if i not in board.legal_moves:
                    legal_logits[board_idx][i] = -1e8
        return legal_logits, logits, values
    
    def _process_logits(self, logits: torch.tensor) -> torch.tensor:
        if self.mode == 'inference':
            # argmax
            result = torch.argmax(logits, dim=1) 
        elif self.mode == 'training':
            # sample
            probs = torch.nn.functional.softmax(logits, dim=1)
            result = torch.multinomial(probs, num_samples=1)
        return result

    def train(self):
        self.mode = 'training'

    def eval(self):
        # self.optimizer.zero_grad()
        self.mode = 'inference'

    def choose(self, boards, return_logits:bool = False) -> torch.tensor:
        if self.ply == 0:
            legal_logits, logits, values = self._get_legal_logits(boards=boards)
            result = self._process_logits(logits=legal_logits)
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

        board_tensor = self._boards_to_tensor(boards=[board])[0]
        processed_reward = self._preprocess_reward(reward=reward)

        self.buffer.add(
            observation=board_tensor,
            action=action,
            reward=processed_reward,
            logits=logits.detach(),
            value=value.detach(),
            game_idx=game_idx,
        )

    def update(self):
        # 
        self.buffer.flatten_buffers()

        buffer_dataloader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
        for batch_idx, (observations, actions, logits, advantages, returns) in enumerate(buffer_dataloader):
            self.optimizer.zero_grad()
            ## preprocessing
            advantages = advantages.squeeze(1)
            returns = returns.unsqueeze(1).float().to(self.device)

            # print(observations.shape, observations.device, observations.dtype)
            # print(actions.shape, actions.device, actions.dtype)
            # print(logits.shape, logits.device, logits.dtype)
            # print(advantages.shape, advantages.device, advantages.dtype)
            # print(returns.shape, returns.device, returns.dtype)

            # 
            predictions, values = self.net(observations)


            # print(predictions.shape, predictions.device, predictions.dtype)
            # print(values.shape, values.device, values.dtype)
            # input()

            loss = self._loss_fn(predictions, observations, actions, logits, advantages, values, returns)

            loss.backward()
            self.optimizer.step()

            logger.debug(f'Batch {batch_idx} completed with loss: {loss.item():.6f}')

        self.buffer.reset()


class TDAgent:
    def __init__(
            self, 
            ply: int,
            net,
            games_per_iter: int,
            device,
            mode: str = 'training',
        ):
        self.learning_rate = 1e-5
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.games_per_iter = games_per_iter
        self.ply = ply
        self.mode = mode
        self.device=device
        
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.buffer = TDBuffer(games_per_iter=self.games_per_iter,gamma=1.0)

        # summary_str = summary(self.net, input_size=(self.batch_size, 4, 4, 17))
        # model_summary_str = '\n'+str(summary_str)
        # logger.info(model_summary_str)
        # logger.info(f'Agent trains for epochs {self.epochs} at batch size: {self.batch_size} ')

    def _loss_fn(self, 
            returns: torch.tensor,
            values: torch.tensor,
        ):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(values, returns)
        return loss
    
    def _preprocess_reward(self, reward: float):
        if reward <= 0: return 0
        return log(reward, 2)

    def _boards_to_tensor(self, boards) -> torch.tensor:
        '''takes a list of n boards and makes an n,17 tensor with tile "tokens"'''
        state_tensor = torch.zeros((len(boards), 17), dtype=torch.int).to(self.device)
        for board_idx, board in enumerate(boards):
            state_tensor[board_idx][0] = 17 # class token first
            for i in range(4):
                for j in range(4):
                    state_tensor[board_idx][(4*i)+j+1] = int(board[i][j])
        return state_tensor
    
    def _expectimax_helper(self, board, depth):
        # if game over
        if len(board.legal_moves) == 0:
            return 0.0, 0
        # make afterstates
        afterstates = []
        for move in board.legal_moves:
            new_board = Board(board.board.copy())
            # NOTE this is just the internal board, not the board object
            afterstates.append(new_board.move_tiles(move)[0])

        leaf_nodes = []
        afterstate_values = []
        for afterstate in afterstates:
            if depth == 0:
                leaf_nodes.append(afterstate)
            else:
                empty_cells = []
                for i in range(4):
                    for j in range(4):
                        if afterstate[i][j] == 0:
                            empty_cells.append((i,j))
                afterstate_value = 0
                for empty_cell_i, empty_cell_j in empty_cells:
                    # 2 spawns at 90% prob
                    new_board = afterstate.copy()
                    new_board[empty_cell_i][empty_cell_j] = 1
                    afterstate_value += 0.9*self._expectimax_helper(
                        Board(given_board=new_board),
                        depth-1,
                    )[0]
                    # 4 spawns at 10% prob
                    new_board = afterstate.copy()
                    new_board[empty_cell_i][empty_cell_j] = 2
                    afterstate_value += 0.1*self._expectimax_helper(
                        Board(given_board=new_board),
                        depth-1,
                    )[0]
            
                afterstate_values.append(afterstate_value)

        if depth == 0:
            with torch.no_grad():
                leaf_nodes_tensor = self._boards_to_tensor(leaf_nodes).to(self.device)
                afterstate_values = self.net(leaf_nodes_tensor)
        else:
            afterstate_values = torch.tensor(afterstate_values)

        # take the maximum
        values, indices = torch.max(afterstate_values, dim=0)

        return values.item(), indices.item()

    def _expectimax(self, board):
        _, indices = self._expectimax_helper(
            board,
            depth=self.ply,
        )
        return indices


    def train(self):
        self.mode = 'training'

    def eval(self):
        # self.optimizer.zero_grad()
        self.mode = 'inference'

    def choose(self, boards) -> torch.tensor:
        # TODO this needs work
        moves_to_play = []
        for board in boards:
            highest_value_move_idx = self._expectimax(board)
            move_to_play = board.legal_moves[highest_value_move_idx]
            moves_to_play.append(move_to_play)
        # this needs to be a tensor i think
        moves_to_play = torch.tensor(moves_to_play)
        return moves_to_play

    def add(self,
            board, 
            reward: torch.tensor, 
            game_idx: int = 0
        ):

        board_tensor = self._boards_to_tensor(boards=[board])[0]
        processed_reward = self._preprocess_reward(reward=reward)

        self.buffer.add(
            observation=board_tensor,
            reward=processed_reward,
            game_idx=game_idx,
        )

    def update(self):
        # 
        self.buffer.flatten_buffers()

        buffer_dataloader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
        for batch_idx, (observations, returns) in enumerate(buffer_dataloader):
            self.optimizer.zero_grad()
            ## preprocessing
            returns = returns.unsqueeze(1).float().to(self.device)
            values = self.net(observations)
            loss = self._loss_fn(values, returns)
            loss.backward()
            self.optimizer.step()

            logger.debug(f'Batch {batch_idx} completed with loss: {loss.item():.6f}')

        self.buffer.reset()


if __name__ == '__main__':

    pass
