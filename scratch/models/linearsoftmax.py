import numpy as np
from math import sqrt

from functions import *

class LinearSoftmax:
    def __init__(self, dims, seed = None):
        if seed is not None: np.random.seed(seed)
        self.dims = dims
        self.weights = np.random.normal(loc=0, scale=(1/sqrt(dims[0])), size=(self.dims))

    def forward(self, activation):
        return softmax(np.dot(self.weights, activation))

    def backward(self, state, label, learning_rate):
        activation = self._forward(state)
        gradient = np.zeros(dims)
        gradient[label] = state.transpose()
        gradient -= np.dot(activation, state.transpose())
        self.weights -= learning_rate*gradient
