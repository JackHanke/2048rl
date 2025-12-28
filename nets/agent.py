from numpy import np
import torch
from math import log

def state_to_tensor(state_array: np.array) -> torch.tensor:
    return_tensor = torch.zeros((1, 16, 17))
    for i in range(4):
        for j in range(4):
            return_tensor[0][(4*i)+j][state_array[i][j]] = 1
    return return_tensor

class Agent:
    def __init__(
            self, 
            net, 
            ply: int = 0, 
            mode: str = 'inference'
        ):
        self.net = net
        self.ply = ply
        self.mode = mode

    def choose(self, board):
        state_tensor = state_to_tensor(board.board)
        if self.ply == 0:
            logits, val = self.net(state_tensor)

            # legal filter
            for i in range(4):
                if i not in board.legal_moves:
                    logits[i] = -float('inf')


            if self.mode == 'inference':
                # argmax
                probs = torch.argmax(logits)
                return torch.multinomial(probs, num_samples=1)
            elif self.mode == 'training':
                # sample
                return torch.nn.functional.softmax(logits)

        else:
            # TODO expectimax
            pass
