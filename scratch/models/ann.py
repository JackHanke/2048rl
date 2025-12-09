import numpy as np
import random
from math import sqrt
from functions.activations import *

# code based on Michael Nielsen's Online Textbook: http://neuralnetworksanddeeplearning.com/
class ArtificialNeuralNetwork:
    # dims is tuple of length >=2 that defines the model dimensions
    #   ie. (784, 15, 10) means a 784 x 15 array and a 15 x 10 array 
    # activations is tuple of tuples of vectorized activation functions and their derivatives
    # loss is a tuple of a loss function and its derivative that accepts an activation vector and label vector
    def __init__(self, dims, activation_funcs, seed=None):
        self.activation_funcs = [-1,-1] + activation_funcs # insert filler to align indexing with textbook
        if seed is not None: np.random.seed(seed)
        self.weights = [-1,-1] # insert filler to align indexing with textbook
        self.biases = [-1,-1]
        for dim_index in range(len(dims)-1):
            self.weights.append(np.random.normal(loc=0, scale=(1/sqrt(dims[0])), size=(dims[dim_index+1], dims[dim_index])))
            self.biases.append(np.random.normal(loc=0, scale=(1/sqrt(dims[0])), size=(dims[dim_index+1], 1)))
        self.num_layers = len(dims)

    # forward pass
    def forward(self, activation, include=False):
        weighted_inputs = [-1, activation]
        activations = [-1, activation]
        for layer_index in range(2,self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            weighted_inputs.append(weighted_input)
            activation = self.activation_funcs[layer_index][0](weighted_input)
            activations.append(activation)
        if include: return activation, weighted_inputs, activations
        else: return activation

    # forward and backward pass
    def backward(self, activation, label, learning_rate):            
        # print(f'state = {activation.transpose()}')
        
        # forward pass
        activation, weighted_inputs, activations = self.forward(activation, include=True)
        # backward pass
        # final layer
        if activation.shape == (1,1):
            delta = activation
        else:
            # delta = np.zeros(activation.shape)
            # delta[label] = 1
            delta = np.multiply(self.activation_funcs[-1][1](weighted_inputs[-1]), activation)
        #remaining layers
        for layer_index in range(self.num_layers, 1, -1):
            # compute product before weights change
            product = np.dot(self.weights[layer_index].transpose(), delta)
            m = activations[layer_index-1].shape[1] # batch_size
            weight_gradient = (np.dot(delta, activations[layer_index-1].transpose()))*(1/m) # average weight gradient
            w_grad_norm = np.linalg.norm(weight_gradient)
            bias_gradient = (delta).mean(axis=1, keepdims=True) # average bias gradient
            b_grad_norm = np.linalg.norm(bias_gradient)

            # TODO fix gradient clipping alg below
            # if w_grad_norm < 10: 
            #     print('w grad too big')
            #     self.weights[layer_index] -= learning_rate*weight_gradient
            # else: self.weights[layer_index] -= learning_rate*(weight_gradient/w_grad_norm)
            # if b_grad_norm < 10: 
            #     print('b grad too big')
            #     self.biases[layer_index] -= learning_rate*bias_gradient
            # else: self.biases[layer_index] -= learning_rate*(bias_gradient/b_grad_norm)

            self.weights[layer_index] -= learning_rate*weight_gradient
            self.biases[layer_index] -= learning_rate*bias_gradient

            # print(f'largest w at {layer_index} = {np.amax(self.weights[layer_index])}')
            # print(f'largest w at {layer_index} = {np.amax(self.biases[layer_index])}')
            # print(f'size of w gradient at {layer_index} = {np.linalg.norm(weight_gradient)}')
            # print(f'size of b gradient at {layer_index} = {np.linalg.norm(bias_gradient)}')
            # print(f'norm of weights at {[layer_index]} {np.linalg.norm(self.weights[layer_index])}')
            # print(f'norm of biases at {[layer_index]} {np.linalg.norm(self.biases[layer_index])}')
            # if np.isnan(np.linalg.norm(self.weights[layer_index])):
            #     input()

            # computes (layer_index - 1) delta vector
            if layer_index != 2: delta = np.multiply(product, self.activation_funcs[layer_index-1][1](weighted_inputs[layer_index-1]))


    def save(self):
        pass
