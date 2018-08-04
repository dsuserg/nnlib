#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing perceptron model on xor problem

Created on Tue Jul 31 12:42:57 2018

@author: dsu
"""
import copy
import time
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc


inp = nn.Dataset([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

out = nn.Dataset([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])

tr_dt = [inp, out]     

counter = 0

while(True):
    NN = perc.NPecrep([2, 2, 2, 2, 1], perc.sgmoidFunc)   #trying reinitialize
    counter += 1
    print("attempt № ", counter )
    stime = time.time() 
    NN.train(tr_dt, perc.Backpropagation(100000,1))
    print("training time", time.time() - stime)
    nn_out = np.array([NN.predict(i) for i in inp])
    error = nn.Metric(out,nn_out).standard_deviation()
    print("Standard deviation error: ", error)
    # because sometimes it may stuck in local minima, so there is two ways:
    # reinitialize network to randomize start weights - more preferable
    # continue training 
    if (error < 0.02):
        break
    
head = "{0:^8} | {1:^12}".format("real_out", "predicted")
head_len = len(head)
print(head)
print("-" * head_len)

for i in range(len(out)):
    print("{0:^8} | {1:^12}".format(str(out[i]), str(nn_out[i])))


    print("-" * head_len)
print("Standard deviation error: ", error)
#
#NN2 = copy.deepcopy(NN)
#NN2.set_new_order(2)
#NN.train(tr_dt, perc.Backpropagation_nn(5000,1))
#nn_out = np.array([NN.predict(i) for i in inp])
#error = nn.Metric(out,nn_out).standard_deviation()
#print("Standard deviation error: ", error)
#
#NN.train(tr_dt, perc.Backpropagation(5000,1))
#nn_out = np.array([NN.predict(i) for i in inp])
#error = nn.Metric(out,nn_out).standard_deviation()
#print("Standard deviation error: ", error)

