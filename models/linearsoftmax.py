# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np
import random
from math import sqrt

from functions import *

class LinearSoftmax:
    def __init__(self, dims):
        np.random.seed(1)
        self.dims = dims
        self.weights = np.random.normal(loc=0, scale=(1/dims[0]), size=(self.dims))

    def _forward(self, activation):
        return softmax(np.dot(self.weights, activation))

    def _backward(self, state, label, learning_rate):
        activation = self._forward(state)
        gradient = np.zeros(dims)
        gradient[label] = state.transpose()
        gradient -= np.dot(activation, state.transpose())
        
        self.weights -= learning_rate*gradient
