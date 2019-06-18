#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract classes library

Created on Sat Jun  9 16:02:28 2018

@author: dsu
"""

import numpy as np
import shelve 


class ActivationFunc:
    
    def __init__(self, activationFunc, derActivationFunc):
        self.func = activationFunc
        self.derivative = derActivationFunc
        

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
    
    def calc_neurons_state(self,prev_layer):
        raise NotImplementedError('action must be defined!')
    
    def dtype(self):
        return self._dtype
    
    def __len__(self):
        return self._neurons_cnt
    
#Accumulative class    
class NeuralNetwork:
                  
    def __init__(self, structure, activation_func, dtype = np.float64 ):
        self._dtype = dtype
        self._structure = structure
        self._input_layer = InputLayer(structure[0])
        self._hidden_layers_cnt = len(structure) - 1
        self._layers = []
        self._activation_func = activation_func
    
    #strategy pattern
    #updating coefficients relies on Trainer class                                     
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
        
    def get_activation_func(self):
        return self._activation_func
    
    def get_structure(self):
        return self._structure
        
    def save_network(self, name):
        with shelve.open("Networks_db") as db:
            db[name] = self
    
    @staticmethod
    def load_network(name):
        with shelve.open("Networks_db") as db:
            return db[name]
     
    def dtype(self):
        return self._dtype
        
    def __len__(self):
        return self._hidden_layers_cnt
    
    def __getitem__(self, key):
        return self._layers[key]

    def __iter__(self):
        return (layer for layer in self._layers)
    
#so abstract
class Trainer:
    def __init__(self, *params, **kparams):
        ...
        
    def train(self, neural_network, training_dataset):
        ...
        