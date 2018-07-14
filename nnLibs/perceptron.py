#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 19:32:10 2018

@author: dsu
"""
import numpy as np
import nnLib as nn
import math


def sigmoid(net):
    return(1/(1+math.exp(-net)))

def sigmoid_derivative(sigm,a):
    return a*sigm*(1-sigm)

sgmoidFunc=nn.ActivationFunc(sigmoid,sigmoid_derivative)
        
        
class HiddenLayer(nn.HiddenLayer):       
    def __init__(self,neuronsInLayer,connectedWith):
        nn.HiddenLayer.__init__(self,neuronsInLayer,connectedWith)
        self.order=1
        
    def calcNeuronsState(self):
        order=self.order
        
        for i in range(order):
            self._neurons+=np.dot(self._connectedWith.getNeurons()**(i+1),
                                 self._neuronsCoefficients.getCoeff()[i][1:,])
            self._neurons+=self._neuronsCoefficients.getCoeff()[i][0,:]            #with bias
            
            print(self._connectedWith.getNeurons()**(i+1))
            print(self._neuronsCoefficients.getCoeff()[i][1:,])
            print(self._neuronsCoefficients.getCoeff()[i][0,:])

#            print(self._connectedWith.getNeurons()**(i+1))
#            
#            print(self._neuronsCoefficients.getCoeff()[i])
#            print(self._neuronsCoefficients.getCoeff()[i][1:,])
#            print(self._neuronsCoefficients.getCoeff()[i][0,:])
            
            
    def setNewOrder(self,order):
        self.order=order
        
class NeuronsCoefficients(nn.NeuronsCoefficients):
    def __init__(self,prevLayer,curntLayer):
        self._coefficients=[]
        self._coefficients.append(np.random.rand(prevLayer.neuronsCount()+1,    #with bias
                           curntLayer.neuronsCount()))
    
    def getCoeff(self):
        return self._coefficients
    
class NPecrep(nn.NeuralNetwork):
    def __init__(self,TrainingDataset,structure,activationFunc):
        nn.NeuralNetwork.__init__(self,TrainingDataset,structure,activationFunc)
    
        self._layers.append(HiddenLayer(structure[0],self._inputLayer))
        
        for i in range(1, self._hiddenLayersCnt):
            self._layers.append(HiddenLayer(structure[0],self._layers[i-1]))
        
        [layer.initCoefficients(NeuronsCoefficients) for layer in self._layers]
        
        


a=[[0.1,0.2]]
NN=NPecrep(a,[2],sgmoidFunc)
out=NN.calcNetwork(a[0])
NN.getCoefficients()










#class NPneuralNetwork(nn.NeuralNetwork):
     
    
#self.order=curntLayer.order
#def getOrder(self):
#    return self.order
#    
#def increaseOrder(self,num):
#    self.order+=num