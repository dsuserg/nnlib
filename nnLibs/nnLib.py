#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:02:28 2018

@author: dsu
"""
import numpy as np


class ActivationFunc:
    def __init__(self,activationFunc,derActivationFunc):
        self.func=activationFunc
        self.derivative=derActivationFunc
    
            
class Dataset:
    def __init__(self,data):
        self.data=data
    
    def normalisationLinear(self):
        ...
    
    def normalisationNonlinear(self):
        ...
    
    def denormalisationLinear(self):
        ...
    
    def denormalisationNonlinear(self):
        ...
    
    
class Metric:
    def __init__(self,reqOut,nnOut):
        self.reqOut=reqOut
        self.nnOut=nnOut
        

class NeuronsCoefficients:
    def __init__(self,prevLayer,crntLayer):
        self._coefficients=[]
        self._coefficients.append(np.random.rand(prevLayer.neuronsCount(),
                           crntLayer.neuronsCount()))
    
    def updateCoefficients(self,trainingFunc):                                 #реализуется индивидуально для каждой НС
        trainingFunc(self._coefficients)
            
    def getCoefficients(self):
        return self._coefficients
        
class InputLayer:
    def __init__(self,data):
        self._neuronsInLayer=len(data)
        self._neurons=np.array(data)
    
    def getNeurons(self):
        return self._neurons
    
    def nextExample(self,data):
        self._neurons=data
            
    def neuronsCount(self):
        return self._neuronsInLayer
    
    
    
class HiddenLayer:
    def __init__(self,neuronsInLayer,connectedWith):
        self._neuronsInLayer = neuronsInLayer
        self._neurons = np.array([0.0 for i in range(neuronsInLayer)])
        self._connectedWith = connectedWith
        
    def initCoefficients(self,NeuronsCoefficients):
        self._neuronsCoefficients = NeuronsCoefficients(self._connectedWith,self)
        
    def activateLayer(self,ActivationFunc):
        for i in range(self._neuronsInLayer):
            self._neurons[i] = ActivationFunc.func(self._neurons[i])
            
    def getCoefficients(self):
        return self._neuronsCoefficients.getCoefficients()
    
    def getNeurons(self):
        return self._neurons
    
    def neuronsCount(self):
        return self._neuronsInLayer
    
    def calcNeuronsState(self,prevLayer):
        raise NotImplementedError('action must be defined!')
    
    
class NeuralNetwork:                  
    def __init__(self,TrainingDataset,structure,activationFunc):
        self._hiddenLayersCnt = len(structure)
        self._TrainingDataset = TrainingDataset
        self._layers = []
        self._activationFunc = activationFunc
        self._inputLayer = InputLayer(self._TrainingDataset[0])
                                    
    def getCoefficients(self):
        for layer in self._layers:
            print(layer.getCoefficients())
    
    def calcNetwork(self,data):                                                #possible realisation
        nnOut=[]
        for layer in self._layers:
            layer.calcNeuronsState()
            layer.activateLayer(self._activationFunc)
        nnOut=self._layers[-1].getNeurons()
        return nnOut
    
    def saveNetwork(self,saveAs):
        ...

    
    