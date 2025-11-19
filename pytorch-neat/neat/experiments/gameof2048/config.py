import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet

from neat.experiments.gameof2048.gameof2048 import Gameof2048

class NEATAgent:
    def __init__(self, phenotype, device):
        self.name = 'NEATAgent'
        self.type = 'offline'
        self.version_num = 0
        self.phenotype = phenotype
        self.device = device

    def choose(self, state, afterstates):
        # TODO state to neural net friendly representation

        state_arr = np.ndarray.flatten(state)
        
        stateTensor = torch.unsqueeze(torch.Tensor(state_arr).to(self.device),0)

        predictionTensor = self.phenotype(stateTensor)

        # only choose among legal moves
        for move in range(4):
            if move not in [tup[0] for tup in afterstates]:
                predictionTensor[0][move] = float('-inf')
        
        chosen_action = torch.argmax(predictionTensor)

        return chosen_action

    def update(self, **kwargs): return
        

class Gameof2048Config:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 16
    NUM_OUTPUTS = 4
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 3.9

    POPULATION_SIZE = 500
    NUMBER_OF_GENERATIONS = 10000
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

    NUM_TEST_GAMES = 100

    def fitness_fn(self, genome):
        phenotype = FeedForwardNet(genome, self)
        agent = NEATAgent(phenotype=phenotype, device=self.DEVICE)

        game = Gameof2048(agent=agent)

        final_score = game.play()
        return final_score
    
