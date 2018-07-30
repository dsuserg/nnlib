#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:02:28 2018

@author: dsu
"""
import numpy as np

class ActivationFunc:
    
    def __init__(self, activationFunc, derActivationFunc):
        self.func = activationFunc
        self.derivative = derActivationFunc
    
            
class Dataset:
    
    def __init__(self, data, dtype = np.float64):
        self._data = np.array(data, dtype = dtype)
        self._size = len(self._data)                  
        self._example_size = len(self._data[0])
        self._status = "pure"
        
    def normalisation_linear(self):
        ...
    
    def normalisation_nonlinear(self):
        ...
    
    def denormalisation_linear(self):
        ...
    
    def denormalisation_nonlinear(self):
        ...
    
    def get_status(self):
        return self._status
    
    def example_size(self):
        return self._example_size
    
    def _prepare_data(self):
        self._mins = np.amin(self._data, axis = 0)
        self._maxs = np.amax(self._data, axis = 0)
        self._centers = [(self._maxs[i]+self._mins[i])/2 
                         for i in range(self._example_size)] 
        
    def __getitem__(self, key):
        return self._data[key]
    
    def __len__(self):
        return self._size
       
    def __iter__(self):
        return (example for example in self._data)
    
class Metric:
    
    def __init__(self, req_out, nn_out):
        self._req_out = req_out
        self._nn_out = nn_out
        
    def standard_deviation(self):
        squares_sum = sum((self._req_out - self._nn_out)**2)
        squares_sum /= len(self._req_out)*self._req_out.example_size()
        return squares_sum**(1/2)
        
class InputLayer:
    
    def __init__(self, neurons_cnt):
        self._neurons_cnt = neurons_cnt
    
    def get_neurons(self):
        return self._neurons
    
    def set_example(self,data):
        self._neurons = data
            
    def __len__(self):
        return self._neurons_cnt
    
    
class HiddenLayer:
    
    def __init__(self,neurons_cnt , connected_with, dtype = np.float64):
        self._dtype = dtype
        self._neurons_cnt = neurons_cnt
        self._state_neurons = np.zeros(neurons_cnt, dtype = self._dtype)
        self._neurons = np.zeros(neurons_cnt, dtype = self._dtype)
        self._connected_with = connected_with
        
    def init_сoefficients(self):
        self._neurons_coefficients = []
        self._neurons_coefficients.append(
                np.random.rand(len(self._connected_with),
                               self._neurons_cnt).astype(self._dtype)
                               )
        
    def activate_layer(self, activation_func):
        for i in range(self._neurons_cnt):
            self._state_neurons[i] = self._neurons[i]
            self._neurons[i] = activation_func.func(self._neurons[i])
            
    def get_coefficients(self):
        return self._neurons_coefficients
    
    def get_neurons(self):
        return self._neurons
    
    def get_state_neurons(self):
        return self._state_neurons
    
    def get_connected_layer(self):
        return self._connected_with
    
    def __len__(self):
        return self._neurons_cnt
    
    def calc_neurons_state(self,prev_layer):
        raise NotImplementedError('action must be defined!')
    
    def dtype(self):
        return self._dtype
    
    
class NeuralNetwork:                  
    def __init__(self, structure, activation_func, dtype = np.float64 ):
        self._dtype = dtype
        self._structure = structure
        self._input_layer = InputLayer(structure[0])
        self._hidden_layers_cnt = len(structure) - 1
        self._layers = []
        self._activation_func = activation_func
                                        
    def train(self, training_dataset, trainer):
        trainer.train(self, training_dataset)
        
    def predict(self, data):
        self._input_layer.set_example(np.array(data, copy=False))
        for layer in self._layers:
            layer.calc_neurons_state()
            layer.activate_layer(self._activation_func)
        nn_out = self._layers[-1].get_neurons()
        return np.copy(nn_out)
    
    def get_coefficients(self):
        coeff = []
        for layer in self._layers:
            coeff.append(layer.get_coefficients())
        
        return coeff    
    
    def printo(self):
        coeff = []
        for layer in self._layers:
            coeff.append([layer.get_biases()])
            coeff.append(layer.get_coefficients())
        
        for c in coeff:
            for r in c:
                for k in r:
                    for c in k:
                        print (c)
            print('-'*20)
        
        
    def get_activation_func(self):
        return self._activation_func
    
    def get_structure(self):
        return self._structure
    
    def __len__(self):
        return len(self._structure) - 1
        
    def save_network(self,save_as):
        ...
    
    def dtype(self):
        return self._dtype
        
    def __getitem__(self, key):
        return self._layers[key]

    def __iter__(self):
        return (layer for layer in self._layers)
    
    
class Trainer:
    def __init__(self, *params, **kparams):
        ...
        
    def train(self, neural_network, training_dataset):
        ...
        