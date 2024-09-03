import numpy as np
# define activation functions
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_prime(x): 
    sig = sigmoid(x)
    return sig * (1-sig)

def softmax(x):
    f = np.exp(x - np.max(x))  # shift values
    return f / f.sum(axis=0)

def relu(x): return x * (x > 0)
def relu_prime(x): return 1. * (x > 0)

def leaky_relu(x, m=0.05): return x * (x > 0) + m*x * (x <= 0)
def leaky_relu_prime(x, m=0.05): return 1 * (x > 0) + m * (x <= 0)

