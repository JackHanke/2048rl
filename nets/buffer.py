import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class Buffer(Dataset):
    def __init__(self, games_per_iter:int, gamma: float = 0.99, lam: float = 0.95):
        self.games_per_iter = games_per_iter
        self.gamma = gamma
        self.lam = lam
        self.reset()
        logger.info(f"Buffer initialized with γ={self.gamma}, λ={self.lam}")

    def __len__(self):
        return len(self.return_buffer)
    
    def __getitem__(self, idx):
        return self.observation_buffer[idx], self.action_buffer[idx], self.logits_buffer[idx], self.advantage_buffer[idx], self.value_buffer[idx], self.return_buffer[idx]

    def _discounted_cumulative_sums(self, arr: list[int], coeff: float):
        results = []
        partial_sum = 0
        for value in reversed(arr):
            partial_sum = value + (partial_sum*coeff)
            results.append(partial_sum)
        return reversed(results)
    
    def reset(self):
        # observed buffers
        self.observation_buffer = [[] for _ in range(self.games_per_iter)]
        self.action_buffer = [[] for _ in range(self.games_per_iter)]
        self.reward_buffer = [[] for _ in range(self.games_per_iter)]
        self.value_buffer = [[] for _ in range(self.games_per_iter)]
        self.logits_buffer = [[] for _ in range(self.games_per_iter)]
        # derived buffers
        self.advantage_buffer = [[] for _ in range(self.games_per_iter)]
        self.return_buffer = [[] for _ in range(self.games_per_iter)]

    def add(self, 
            observation: torch.tensor, 
            action: torch.tensor, 
            reward: torch.tensor, 
            logits: torch.tensor,
            value: torch.tensor, 
            game_idx: int = 0
        ):
        self.observation_buffer[game_idx].append(observation)
        self.action_buffer[game_idx].append(action)
        self.reward_buffer[game_idx].append(reward)
        self.value_buffer[game_idx].append(value)
        self.logits_buffer[game_idx].append(logits)

    def finish_trajectory(self, game_idx: int):
        lastValue = 0

        self.reward_buffer[game_idx].append(lastValue*self.gamma)
        self.value_buffer[game_idx].append(lastValue)

        deltas = []
        for move_idx, reward in enumerate(self.reward_buffer[game_idx]):
            if move_idx != len(self.reward_buffer[game_idx])-1:
                # TD error
                delta = reward - (self.value_buffer[game_idx][move_idx] - \
                                  (self.value_buffer[game_idx][move_idx+1]*self.gamma))
                deltas.append(delta)

        self.return_buffer[game_idx].extend(self._discounted_cumulative_sums(self.reward_buffer[game_idx], self.gamma))
        self.return_buffer[game_idx].pop() # remove last value from this game's return buffer
        self.advantage_buffer[game_idx].extend(self._discounted_cumulative_sums(deltas, self.gamma*self.lam))

    def flatten_buffers(self):
        '''turn individual game buffers into a single shared buffers'''
        self.observation_buffer = [i for game_buffer in self.observation_buffer for i in game_buffer]
        self.action_buffer = [i for game_buffer in self.action_buffer for i in game_buffer]
        self.reward_buffer = [i for game_buffer in self.reward_buffer for i in game_buffer]
        self.value_buffer = [i for game_buffer in self.value_buffer for i in game_buffer]
        self.logits_buffer = [i for game_buffer in self.logits_buffer for i in game_buffer]
        self.advantage_buffer = [i for game_buffer in self.advantage_buffer for i in game_buffer]
        self.return_buffer = [i for game_buffer in self.return_buffer for i in game_buffer]

if __name__ == '__main__':
    # TODO buffer tests
    pass
