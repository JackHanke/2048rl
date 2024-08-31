import numpy as np
from math import sqrt


class Linear:
    def __init__(self, dims, seed = None):
        if seed is not None: np.random.seed(seed)
        self.dims = dims
        self.weights = np.random.normal(loc=0, scale=(1/sqrt(dims[0])), size=(self.dims))

    def _forward(self, activation):
        return np.dot(self.weights.transpose(), activation)

    def _backward(self, state, label, learning_rate):
        self.weights -= learning_rate*state
