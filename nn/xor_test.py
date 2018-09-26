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
import random
import matplotlib.pyplot as plt


inp = nn.Dataset([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

out = nn.Dataset([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])
cn = 50
inp1 = nn.Dataset([[np.random.rand() for i in range(4)] for j in range(cn)])
out1 = nn.Dataset([[i[2]/sum(i),i[1]/sum(i), i[0]/sum(i), np.sin(sum(i))] for i in inp1])

tinp1 = nn.Dataset([[random.random() for i in range(4)] for j in range(1000)])
tout1 = nn.Dataset([[i[2]/sum(i),i[1]/sum(i), i[0]/sum(i), np.sin(sum(i))] for i in inp1])

#tr_dt = [iNN2np, out]     
tr_dt = [inp1, out1]

counter = 0


n = perc.NPecrep([4, 1, 4], perc.sgmoidFunc)
n2 = copy.deepcopy(n)
n3 = copy.deepcopy(n)


n_out = [n2.predict(i) for i in inp1]
tr_error1 = nn.RMSE(out1, n_out)
print("Training error:", tr_error1)
n2.train(tr_dt, perc.Backpropagation_nn(1000))
n2.set_new_order(2)
n2.train(tr_dt, perc.Speedest_decent(1000, nn.RMSE, 0.01, 0.00001))
n_out = [n2.predict(i) for i in inp1]
tr_error1 = nn.RMSE(out1, n_out)
print("Training error:", tr_error1)

n.train(tr_dt, perc.Backpropagation(2000))
n_out = [n.predict(i) for i in inp1]
tr_error1 = nn.RMSE(out1, n_out)
print("Training error:", tr_error1)

n3.set_new_order(2)
n3.train(tr_dt, perc.Backpropagation_nn(2000))
n_out = [n3.predict(i) for i in inp1]
tr_error1 = nn.RMSE(out1, n_out)
print("Training error:", tr_error1)


print('-'*20)
#
#

#
#n2.train(tr_dt, perc.Backpropagation_nn(10000))
#n_out = [n2.predict(i) for i in inp1]
#tr_error2 = nn.RMSE(out1, n_out)
#print("Training error:", tr_error2)
#
n_out = [n2.predict(i) for i in tinp1]
te_error2 = nn.RMSE(tout1, n_out)
print("Test error:    ", te_error2)

#
#n.train(tr_dt, perc.Backpropagation(10000))
#n_out = [n.predict(i) for i in inp1]
#tr_error1 = nn.RMSE(out1, n_out)
#print("Training error:", tr_error1)
#
n_out = [n.predict(i) for i in tinp1]
te_error1 = nn.RMSE(tout1, n_out)
print("Test error:    ", te_error1)

