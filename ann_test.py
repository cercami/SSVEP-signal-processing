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
from numpy import newaxis as NA

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

        # calculate signals into final output layer
        final_inputs = self.who @ hidden_outputs
        # calculate the signals emerging from final output layer
        final_outputs = self.act_func(final_inputs)
        # output layer error: target - actual
        output_errors = targets - final_outputs
        # hidden layer error is the output errors, split by weights, recombined at hidden nodes
        hidden_errors = self.who.T @ output_errors

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * ((output_errors * final_outputs * (1.-final_outputs)) @ hidden_outputs[NA,:].T)
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * ((hidden_errors * hidden_outputs * (1.-hidden_outputs)) @ inputs.T)

        pass


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = self.wih @ inputs

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.act_func(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = self.who @ hidden_outputs
        # calculate the signals emerging from final output layer
        final_outputs = self.act_func(final_inputs)

        return final_outputs

    pass