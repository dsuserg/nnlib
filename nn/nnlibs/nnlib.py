#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract classes library

Created on Sat Jun  9 16:02:28 2018

@author: dsu
"""

import numpy as np
import shelve 

#class Dataset(np.ndarray):
#    
#    def __new__(cls, object, dtype=None, copy=True, 
#                order='K', subok=False, ndmin=0):
#        
#        obj = np.array(object, dtype=None, copy=True, 
#                order='K', subok=False, ndmin=0).view(cls)
#        
#        obj._status = "pure"
#        obj._range = None
#        obj._expert = None
#        obj._percentage = None
#        obj._max_elems = None
#        obj._min_elems = None
#        obj._centers = None
#        
#        return obj
#        
#    def tune_up(self, expert = None, percentage = None):
#        self._max_elems = np.zeros((self._example_size))
#        self._min_elems = np.zeros((self._example_size))
#        self._centers = np.zeros((self._example_size))
#       
#        if expert:
#            for i in range(self._example_size):
#                if expert[i]:
#                    self._min_elems[i] = expert[i][0]
#                    self._min_elems[i] = expert[i][1]
#                else:
#                    self._max_elems[i] = max(self._data[:,i])
#                    self._min_elems[i] = min(self._data[:,i])            
#
#        else:
#            for i in range(self._example_size):
#                self._max_elems[i] = max(self._data[:,i])
#                self._min_elems[i] = min(self._data[:,i])
#        
#        if percentage:
#            for i in range(self._example_size):
#                if percentage[i]:
#                    self._max_elems[i] += percentage[i]*(self._max_elems[i] 
#                                                         - self._min_elems[i])
#                    
#                    self._min_elems[i] -= percentage[i]*(self._max_elems[i] 
#                                                         - self._min_elems[i])
#                else: continue
#       
#        for i in range(self._example_size):
#            self._centers[i] = (self._max_elems[i] + self._min_elems[i])/2  
#        
#        
#    def normalisation_linear(self, rng):
#        if rng[0]:
#            for i in range(self._size):
#                for j in range(self._example_size):
#                    numerator = 2*(self._data[i][j] - self._min_elems[j])
#                    denominator = self._max_elems[j] - self._min_elems[j]
#                    self._data[i][j] = numerator/denominator - 1            
#
#        else:
#            for i in range(self._size):
#                for j in range(self._example_size):
#                    numerator = self._data[i][j] - self._min_elems[j]
#                    denominator = self._max_elems[j] - self._min_elems[j]
#                    self._data[i][j] = numerator/denominator
#                    
#    def normalisation_nonlinear(self, rng, alfa):
#        if rng[0]:
#            for i in range(self._size):
#                for j in range(self._example_size):
#                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
#                    self._data[i][j] = (np.exp(degree)-1)/(np.exp(degree)+1)
#            
#        else:    
#            for i in range(self._size):
#                for j in range(self._example_size):
#                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
#                    self._data[i][j] = 1/np.exp(degree) + 1
#                    
#    def denormalisation_linear(self, rng, example):
#        out = np.zeroes((self._example_size))
#        if rng[0]:
#            for i in range(self._example_size):
#                substraction = self._max_elems[i] - self._min_elems[i]
#                out[i] = self._min_elems[i] + (example[i] + 1)*substraction/2
#        else:
#            for i in range(self._example_size):
#                substraction = self._max_elems[i] - self._min_elems[i]
#                out[i] = self._min_elems[i] + example[i]*substraction
#    
#    def denormalisation_nonlinear(self, rng, alfa, example):
#        out = np.zeroes((self._example_size))
#        if rng[0]:
#            for i in range(self._example_size):
#                out[i] = self._centers[i] - (1/alfa[i])* np.log(1/example[i]-1)
#        else:
#            for i in range(self._example_size):
#                temp = (1-self._centers[i])/(1+self._centers[i])
#                out[i] = self._centers[i] - (1/alfa[i])* np.log(temp)
#    
#    def get_status(self):
#        return self._status
#    
#    def example_size(self):
#        return self._example_size
#    
#    def _prepare_data(self):
#        self._mins = np.amin(self._data, axis = 0)
#        self._maxs = np.amax(self._data, axis = 0)
#        self._centers = [(self._maxs[i]+self._mins[i])/2 
#                         for i in range(self._example_size)] 
#        



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
        self._range = None
        self._expert = None
        self._percentage = None
        self._max_elems = None
        self._min_elems = None
        self._centers = None
        
    def tune_up(self, expert = None, percentage = None):
        self._max_elems = np.zeros((self._example_size))
        self._min_elems = np.zeros((self._example_size))
        self._centers = np.zeros((self._example_size))
       
        if expert:
            for i in range(self._example_size):
                if expert[i]:
                    self._min_elems[i] = expert[i][0]
                    self._min_elems[i] = expert[i][1]
                else:
                    self._max_elems[i] = max(self._data[:,i])
                    self._min_elems[i] = min(self._data[:,i])            

        else:
            for i in range(self._example_size):
                self._max_elems[i] = max(self._data[:,i])
                self._min_elems[i] = min(self._data[:,i])
        
        if percentage:
            for i in range(self._example_size):
                if percentage[i]:
                    self._max_elems[i] += percentage[i]*(self._max_elems[i] 
                                                         - self._min_elems[i])
                    
                    self._min_elems[i] -= percentage[i]*(self._max_elems[i] 
                                                         - self._min_elems[i])
                else: continue
       
        for i in range(self._example_size):
            self._centers[i] = (self._max_elems[i] + self._min_elems[i])/2  
        
        
    def normalisation_linear(self, rng):
        if rng[0]:
            for i in range(self._size):
                for j in range(self._example_size):
                    numerator = 2*(self._data[i][j] - self._min_elems[j])
                    denominator = self._max_elems[j] - self._min_elems[j]
                    self._data[i][j] = numerator/denominator - 1            

        else:
            for i in range(self._size):
                for j in range(self._example_size):
                    numerator = self._data[i][j] - self._min_elems[j]
                    denominator = self._max_elems[j] - self._min_elems[j]
                    self._data[i][j] = numerator/denominator
                    
    def normalisation_nonlinear(self, rng, alfa):
        if rng[0]:
            for i in range(self._size):
                for j in range(self._example_size):
                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
                    self._data[i][j] = (np.exp(degree)-1)/(np.exp(degree)+1)
            
        else:    
            for i in range(self._size):
                for j in range(self._example_size):
                    degree = -alfa[j]*(self.data[i][j] - self._centers[j])
                    self._data[i][j] = 1/np.exp(degree) + 1
                    
                    
    
    def denormalisation_linear(self, rng, example):
        out = np.zeroes((self._example_size))
        if rng[0]:
            for i in range(self._example_size):
                substraction = self._max_elems[i] - self._min_elems[i]
                out[i] = self._min_elems[i] + (example[i] + 1)*substraction/2
        else:
            for i in range(self._example_size):
                substraction = self._max_elems[i] - self._min_elems[i]
                out[i] = self._min_elems[i] + example[i]*substraction
    
    def denormalisation_nonlinear(self, rng, alfa, example):
        out = np.zeroes((self._example_size))
        if rng[0]:
            for i in range(self._example_size):
                out[i] = self._centers[i] - (1/alfa[i])* np.log(1/example[i]-1)
        else:
            for i in range(self._example_size):
                temp = (1-self._centers[i])/(1+self._centers[i])
                out[i] = self._centers[i] - (1/alfa[i])* np.log(temp)
    
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
    
    
def RMSE(req_out,nn_out):
    summ = 0
        
    for i in range(len(req_out)):
        summ += sum((req_out[i] - nn_out[i])**2)
            
    a = len(req_out)*req_out.example_size()
        
    return (summ/a)**(1/2)
    

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
        