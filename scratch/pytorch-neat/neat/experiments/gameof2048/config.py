import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
from math import log


from neat.experiments.gameof2048.gameof2048 import Gameof2048        

class Gameof2048Config:
    EXPERIMENT_NAME = 'gameof2048'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 11*16
    NUM_OUTPUTS = 4
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 10
    FITNESS_SAVE_THRESHOLD = 8.5

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

    def fitness_fn(self, genome):
        phenotype = FeedForwardNet(genome, self)
        agent = NEATAgent(phenotype=phenotype, device=self.DEVICE)

        mean_score = 0
        for _ in range(self.NUM_EVAL_GAMES):
            game = Gameof2048(agent=agent)
            final_score = game.play()

            mean_score += log(final_score)
        
        mean_score = mean_score/self.NUM_EVAL_GAMES

        return mean_score
    
class NEATAgent:
    def __init__(self, phenotype, device):
        self.name = 'NEATAgent'
        self.type = 'offline'
        self.version_num = 0
        self.phenotype = phenotype
        self.device = device

    def choose(self, state, afterstates):
        state_arr = np.ndarray.flatten(state)

        one_hot_arr = np.zeros(11*16)
        for loc, tile in enumerate(state_arr):
            # first 16 are 1 if 0, else 0, next 16 is 1 if the 2 tile, else 0, next 16 for 4 tile, etc.
            tile_index = 0 if tile == 0 else int(log(tile, 2))
            one_hot_arr[16*tile_index + loc] = 1

        stateTensor = torch.unsqueeze(torch.Tensor(one_hot_arr).to(self.device),0)

        predictionTensor = self.phenotype(stateTensor)

        # only choose among legal moves
        for move in range(4):
            if move not in [tup[0] for tup in afterstates]:
                predictionTensor[0][move] = float('-inf')
        
        chosen_action = torch.argmax(predictionTensor)

        return chosen_action

    def update(self, **kwargs): return