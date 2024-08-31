from macht.macht.term import main
from statistics import stdev, median, mean
from time import time
import random
import numpy as np
from model import Network, sigmoid, sigmoid_prime, softmax, LinearSoftmax, relu, relu_prime, leaky_relu, leaky_relu_prime

def simple_exponent_state_rep(grid):
    rep = [0 for _ in range(16)]
    for row_index, row in enumerate(grid):
        for column_index, tile in enumerate(row):
            if tile is None:
                rep[(row_index * 4)  + column_index] = 0
            else:
                rep[(row_index * 4)  + column_index] = tile.exponent/16
    return rep

def one_hot_state_rep(grid):
    # exponent 0 1 2 3 4 5 6 7 8 9
    # encoding 0 0 0 0 0 0 0 0 1 0 
    rep = []
    for row_index, row in enumerate(grid):
        for column_index, tile in enumerate(row):
            if tile is None:
                rep += [1] + [0 for _ in range(15)] 
            else:
                temp_vec = [0 for _ in range(16)]
                temp_vec[tile.exponent] = 1
                rep += temp_vec
    return rep

def mask(activation, invalid_moves):
    pass

