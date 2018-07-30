#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 19:32:10 2018

@author: dsu
"""
import numpy as np
import nnlib as nn
import math
import time

def sigmoid(net):
    return 1/(1 + math.e**(-net))

def sigmoid_derivative(net, a = 1.0):
    sigm = sigmoid(net)
    return a * sigm * (1 - sigm)

sgmoidFunc=nn.ActivationFunc(sigmoid, sigmoid_derivative)
        
        
class HiddenLayer(nn.HiddenLayer):       
    def __init__(self, neuronsInLayer, connectedWith, dtype = np.float64):
        nn.HiddenLayer.__init__(self, neuronsInLayer, connectedWith, dtype)
        self.order = 1
        
    def init_сoefficients(self):
        self._neurons_coefficients = []
        coeff = np.random.rand(len(self._connected_with),            
                               self._neurons_cnt
                               ).astype(self._dtype)
        self._neurons_coefficients.append(coeff)
        
        self._bias_coefficients = []
        biases = np.random.rand(self._neurons_cnt).astype(self._dtype)
        self._bias_coefficients.append(biases)

    def calc_neurons_state(self):
        order = self.order
        
        for i in range(order):
            self._neurons += np.dot(
                          self._connected_with.get_neurons()**(i+1),
                          self._neurons_coefficients[i])
            self._neurons += self._bias_coefficients[i]             #with bias
            
    def set_new_order(self, order):
        if order > self.order:
            zeros = np.zeros([
                              self._connected_with.neurons_сount() + 1,          #with bias
                              self._neurons_cnt
                              ], 
                              dtype = self._dtype
                             )
            for i in range(order - self.order):
                self._neurons_coefficients.append(zeros)
        else:
            self._neurons_coefficients = self._neurons_coefficients[: order - 1]
            
        self.order = order
        
    def get_biases(self):
        return self._bias_coefficients
    
class NPecrep(nn.NeuralNetwork):
    def __init__(self, structure, activation_func, dtype = np.float64):
        nn.NeuralNetwork.__init__(self, structure, activation_func, dtype)
        
        structure=structure[1:]
        
        fst_hidden = HiddenLayer(structure[0], self._input_layer, self._dtype)
        
        self._layers.append(fst_hidden)
        
        for i in range(1, len(structure)):
            hidden = HiddenLayer(structure[i], self._layers[i-1] , self._dtype)
            self._layers.append(hidden)
        
        [layer.init_сoefficients() for layer in self._layers]
        
        
class Backpropagation(nn.Trainer):
    def __init__(self, epoches_cnt, train_speed = 1.0):
        self.epoches_cnt = epoches_cnt
        self.train_speed = train_speed
        
     
    def calc_discrepancies(self, discrepancies, errors, network):
        n_structure = network.get_structure()
        derivative = network.get_activation_func().derivative
        
        for i in range(n_structure[-1]):
            discrepancies[-1][i] = (derivative(network[-1].get_state_neurons()) 
                                    * errors[i])
            
        for i in range(len(network) - 2 , -1, -1):
            derivatives = derivative(network[i].get_state_neurons())
            weights = network[i+1].get_coefficients()[0]
            discrepancies[i] = (np.dot(discrepancies[i + 1], 
                                       np.transpose(weights)) 
                                * derivatives)
       
    def update_coefficients(self, discrepancies, example, network, tr_speed):
        #for 1st layer special rules
        weights = network[0].get_coefficients()[0]
        weights += tr_speed*discrepancies[0]*example[:,np.newaxis]
        for i in range(1, len(network)):
            weights = network[i].get_coefficients()[0]
            weights += tr_speed*discrepancies[i]*(network[i-1].get_neurons()[:,np.newaxis])
            
        for i in range(len(network)):
            biases = network[i].get_biases()[0]
            biases += tr_speed*discrepancies[i]
 
    def train(self, neural_network, training_dataset):
        network = neural_network
        n_structure = network.get_structure()
        dtype = network.dtype()
        inp = training_dataset[0]
        req_out = training_dataset[1]
        nn_out = np.array(req_out)
        exmpls_cnt = len(inp)
        discrepancies = [np.zeros(i, dtype) for i in n_structure[1:]]
 
        
        for i in range(self.epoches_cnt):
            for j in range(exmpls_cnt):
                out = network.predict(inp[j])
                nn_out[j] = out
                errors = req_out[j] - nn_out[j]  
                self.calc_discrepancies(discrepancies, errors, neural_network)
                self.update_coefficients(discrepancies, 
                                         inp[j], 
                                         neural_network,
                                         self.train_speed)
#            print(nn.Metric(req_out,nn_out).standard_deviation(),i)

          
inp = nn.Dataset([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

out = nn.Dataset([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])
 
tr_dt = [inp, out]               
NN = NPecrep([2, 10, 1], sgmoidFunc)
stime=time.time()  
NN.train(tr_dt, Backpropagation(100000,1))
print(time.time()-stime)
nn_out=np.array([NN.predict(i) for i in inp])
print(nn.Metric(out,nn_out).standard_deviation())

