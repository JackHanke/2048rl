import json
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
# from math import log

# from neat.experiments.gameof2048.gameof2048 import Gameof2048        

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

def trinary_board_rep(board_str: str):
    bool_board = [0 for _ in range(9)]

    for idx, val in enumerate(board_str):
        if val == 'X':
            bool_board[idx] = 1

    for idx, val in enumerate(board_str):
        if val == 'O':
            bool_board[idx] = -1

    return bool_board

class tttConfig:
    with open("all_states.json", "r") as file:
        states_dict = json.load(file)

    X, Y = [], []
    for board_str, moves in states_dict.items():
        # binary_board = binary_board_rep(board_str=board_str)
        binary_board = trinary_board_rep(board_str=board_str)
        X.append(binary_board)
        # Y.append(moves[0]) # TODO this needs to be fixed
    states = [val for val in states_dict.values()]

    EXPERIMENT_NAME = 'ttt'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    X_data = torch.tensor(X).float().to(DEVICE)
    y_data = torch.tensor(Y).long().to(DEVICE)

    NUM_INPUTS = 9
    NUM_OUTPUTS = 9
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 100
    FITNESS_SAVE_THRESHOLD = 50

    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 1500
    # POPULATION_SIZE = 2
    # NUMBER_OF_GENERATIONS = 2

    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    # number of games to calculate fitness
    NUM_EVAL_GAMES = 2
    # number of games to evaluate final solution
    NUM_TEST_GAMES = 10

    def fitness_fn(self, genome, X_data=X_data, states=states):
        phenotype = FeedForwardNet(genome, self).to("cuda:0") # TODO fix

        correct = 0
        for board_idx, row in enumerate(X_data):
            # print(row.shape)
            logits = phenotype(row.unsqueeze(0))
            preds = torch.argmax(logits, dim=1).item()
            if preds in states[board_idx]:
                correct += 1

        return 100*correct/X_data.shape[0]
    
