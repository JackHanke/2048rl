import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# TODO make Buffer handle multiple concurrent games

class Buffer(Dataset):
    def __init__(self, games_per_iter:int, gamma: float = 0.99, lam: float = 0.95):
        self.games_per_iter = games_per_iter
        self.gamma = gamma
        self.lam = lam
        self.reset()
        logger.info(f"Buffer initialized with γ={self.gamma}, λ={self.lam}")

    def __len__(self):
        return
    
    def __getitem__(self):
        return

    def _discounted_cumulative_sums(self, arr: list[int], coeff: float):
        results = []
        partial_sum = 0
        for value in arr.reverse():
            partial_sum = value + (partial_sum*coeff)
            results.append(partial_sum)
        return results.reverse()
    
    def reset(self):
        # 
        self.observationBuffer = [[] for _ in range(self.games_per_iter)]
        self.actionBuffer = [[] for _ in range(self.games_per_iter)]
        self.rewardBuffer = [[] for _ in range(self.games_per_iter)]
        self.valueBuffer = [[] for _ in range(self.games_per_iter)]
        self.logprobabilityBuffer = [[] for _ in range(self.games_per_iter)]
        # 
        self.advantageBuffer = []
        self.returnBuffer = []
        # 
        self.trajectoryStartIndex = 0
        self.pointer = 0

    def add(self, 
            observation: torch.tensor, 
            action: torch.tensor, 
            reward: torch.tensor, 
            value: torch.tensor, 
            logprobability: torch.tensor,
            game_idx: int = 0
        ):
        self.observationBuffer[game_idx].append(observation)
        self.actionBuffer[game_idx].append(action)
        self.rewardBuffer[game_idx].append(reward)
        self.valueBuffer[game_idx].append(value)
        self.logprobabilityBuffer[game_idx].append(logprobability)
        self.pointer += 1

    def finish_trajectory(self):
        lastValue = 0

        rewards = self.rewardBuffer[self.trajectoryStartIndex:self.pointer].append(lastValue*self.gamma)
        values = self.valueBuffer[self.trajectoryStartIndex:self.pointer].append(lastValue)
        deltas = [] # TODO

        self.advantageBuffer.extend(self._discounted_cumulative_sums(deltas, self.gamma*self.lam))
        self.returnBuffer.extend(self._discounted_cumulative_sums(rewards, self.gamma))
        self.trajectoryStartIndex = self.pointer

    def get(self):
        
        # TODO

        return [
            self.observationBuffer,
            self.actionBuffer,
            self.advantageBuffer,
            self.returnBuffer,
            self.logprobabilityBuffer,
        ]
    
    def flatten_buffers(self):
        # TODO flatten buffers
        pass
