import numpy as np
from math import log
# custom implementation representation functions

def log_modified(x, b=2):
    if x == 0: return 0
    return log(x, b)

def identity_rep(grid):
    return grid

def simple_exponent_state_rep(grid):
    rep = np.array([0 for _ in range(16)])
    for row_index, row in enumerate(grid):
        for column_index, val in enumerate(row):
            if val != 0:
                rep[(row_index * 4)  + column_index] = val
    return rep

# macht state representation functions
def simple_exponent_state_rep_macht(grid):
    rep = [0 for _ in range(16)]
    for row_index, row in enumerate(grid):
        for column_index, tile in enumerate(row):
            if tile is None:
                rep[(row_index * 4)  + column_index] = 0
            else:
                rep[(row_index * 4)  + column_index] = tile.exponent
    return rep

def one_hot_state_rep_macht(grid):
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
