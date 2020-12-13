# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
Created on Sun Dec 13 10:22:28 2020
Version: 0.1

A basic three-layer backpropagation neural network demo for MNIST dataset
    
update: 2020/12/12

"""

# %% load in modules
import numpy as np
from numpy.random import normal
from scipy.special import expit

# %% neural network class definition
class neural_network:
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden and output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # link weight matrices, wih and who
        # weight inside the arrays are w_i_i: from i to j in the next layer
        # w11  w21
        # w12  w22, etc
        self.wih = normal(0., pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = normal(0., pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learning_rate

        # activation function: sigmoid function
        self.act_func = lambda x: expit(x)

        pass


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = self.wih @ inputs
        # calculate the signals emerging from hidder layer
        hidden_outputs = self.act_func(hidden_inputs)


        
