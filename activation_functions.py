'''
_ _ _ Activation Functions _ _ _

'''
import numpy as np 

# - - - - - - - - - - - - - - - - - - - - - - - - - -

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

# - - - - - - - - - - - - - - - - - - - - - - - - - -

tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
tanh_derivative = lambda x: 1 - (np.exp(x) - np.exp(-x)) ** 2 / (np.exp(x) + np.exp(-x)) ** 2

# - - - - - - - - - - - - - - - - - - - - - - - - - -

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1 * (x > 0)

# - - - - - - - - - - - - - - - - - - - - - - - - - -

sin = lambda x: np.sin(x)
sin_derivative = lambda x: np.cos(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - -

def softmax(x):
    x -= np.max(x) # <-- helps with numerical stability apparently 
    return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

# - - - - - - - - - - - - - - - - - - - - - - - - - -

def linear(x):
    return x

def linear_derivative(x):
    return 1

# - - - - - - - - - - - - - - - - - - - - - - - - - -

def warp(x, num_dims):
    return np.exp(x - num_dims)

warp_derivative = warp

# - - - - - - - - - - - - - - - - - - - - - - - - - -


