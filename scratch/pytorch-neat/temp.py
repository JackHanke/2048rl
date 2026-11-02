# NOTE delete this file

import json
import pickle
import torch
from time import time

import neat.experiments.ttt.config as c
from neat.phenotype.feed_forward import FeedForwardNet

def binary_board_rep(board_str: str):
    ''' 18 binary digits, first 9 are position of X and second 9 are position of O '''
    bool_board = [0 for _ in range(18)]

    for idx, val in enumerate(board_str):
        if val == 'X':
            bool_board[idx] = 1

    for idx, val in enumerate(board_str):
        if val == 'O':
            bool_board[idx+9] = 1

    return bool_board

# best_genome_path = '/home/jack/vault/software/2048rl/scratch/pytorch-neat/neat/experiments/ttt/checkpoints/exp_2026-04-27 21:34:36.949546_gen_45.pickle'
best_genome_path = '/home/jack/vault/software/2048rl/scratch/pytorch-neat/neat/experiments/ttt/checkpoints/exp_2026-04-27 21:34:36.949546_gen_59.pickle'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(best_genome_path, 'rb') as f:
    best_genome = pickle.load(f)

model = FeedForwardNet(best_genome, c.tttConfig)
num_params = sum(p.numel() for p in model.parameters())

with open("all_states.json", "r") as file:
    states_dict = json.load(file)

X, Y = [], []
for board_str, moves in states_dict.items():
    binary_board = binary_board_rep(board_str=board_str)
    X.append(binary_board)
    # Y.append(moves[0]) # TODO this needs to be fixed
states = [val for val in states_dict.values()]
X_data = torch.tensor(X).float()

start = time()

correct = 0

print(f'X_data.shape: {X_data.shape}')

logits = model(X_data)
print(f'logits.shape: {logits.shape}')
preds = torch.argmax(logits, dim=1)

print(preds.shape)

# for board_idx, row in enumerate(X_data):
#     # print(row.shape)
#     logits = model(row.unsqueeze(0))
#     preds = torch.argmax(logits, dim=1).item()
#     if preds in states[board_idx]:
#         correct += 1

fitness = 100*correct/X_data.shape[0]
print(f'Inference computed in {time()-start:.4f}s')

print(f'Genome length: {len(best_genome.connection_genes)} Parameter count: {num_params}, fitness: {correct} correct, {fitness:.4f}%')
