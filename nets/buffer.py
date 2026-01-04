import torch


class Buffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def _reset(self):
        self.observationBuffer = []
        self.actionBuffer = []
        self.advantageBuffer = []
        self.rewardBuffer = []
        self.returnBuffer = []
        self.valueBuffer = []
        self.logprobabilityBuffer = []
        self.trajectoryStartIndex = 0
        self.pointer = 0

    def _discounted_cumulative_sums(self, arr: list[int], coeff: float):
        results = []
        partial_sum = 0
        for value in arr.reverse():
            partial_sum = value + (partial_sum*coeff)
            results.append(partial_sum)
        return results.reverse()

    def add(self, 
            observation: torch.tensor, 
            action: torch.tensor, 
            reward: torch.tensor, 
            value: torch.tensor, 
            logprobability: torch.tensor
        ):
        self.observationBuffer.append(observation)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.valueBuffer.append(value)
        self.logprobabilityBuffer.append(logprobability)
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
