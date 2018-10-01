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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = nn.Dataset(X_train)
X_train.tune_up()
X_train.normalisation_linear([0,1])

X_test = nn.Dataset(X_test)
X_test.tune_up()
X_test.normalisation_linear([0,1])

y_train = nn.Dataset(y_train[:,np.newaxis])
y_train.tune_up()
y_train.normalisation_linear([0,1])

y_test = nn.Dataset(y_test[:,np.newaxis])
y_test.tune_up()
y_test.normalisation_linear([0,1])


n = perc.NPecrep([4, 4, 1], perc.sgmoidFunc)
n2 = copy.deepcopy(n)
n3 = copy.deepcopy(n)








#inp = nn.Dataset([[0.0, 0.0],
#                  [0.0, 1.0],
#                  [1.0, 0.0],
#                  [1.0, 1.0]])
#
#out = nn.Dataset([[0.0],
#                  [1.0],
#                  [1.0],
#                  [0.0]])
#cn = 50



#inp1 = nn.Dataset([[np.random.rand() for i in range(4)] for j in range(cn)])
#out1 = nn.Dataset([[i[2]/sum(i),i[1]/sum(i), i[0]/sum(i), np.sin(sum(i))] for i in inp1])
#
#tinp1 = nn.Dataset([[random.random() for i in range(4)] for j in range(1000)])
#tout1 = nn.Dataset([[i[2]/sum(i),i[1]/sum(i), i[0]/sum(i), np.sin(sum(i))] for i in inp1])


tr_dt = [X_train, y_train]     



n_out = [n2.predict(i) for i in X_train]
tr_error1 = nn.RMSE(y_train, n_out)
print("Training error:", tr_error1)
#n2.train(tr_dt, perc.Backpropagation_nn(2000))
#n2.set_new_order(2)
#n2.train(tr_dt, perc.Speedest_decent(2000, nn.RMSE, 0.01, 0.00001))
#n_out = [n2.predict(i) for i in X_train]
#tr_error1 = nn.RMSE(y_train, n_out)
#print("Training error:", tr_error1)


n.train(tr_dt, perc.Backpropagation(5000))
n_out = [n.predict(i) for i in X_train]
tr_error1 = nn.RMSE(y_train, n_out)
print("Training error:", tr_error1)

n3.set_new_order(2)
n3.train(tr_dt, perc.Backpropagation_nn(5000))
n_out = [n3.predict(i) for i in X_train]
tr_error1 = nn.RMSE(y_train, n_out)
print("Training error:", tr_error1)


print('-'*20)
#
#

#
#n2.train(tr_dt, perc.Backpropagation_nn(10000))
#n_out = [n2.predict(i) for i in inp1]
#tr_error2 = nn.RMSE(out1, n_out)
#print("Training error:", tr_error2)
##
#n_out = [n2.predict(i) for i in tinp1]
#te_error2 = nn.RMSE(tout1, n_out)
#print("Test error:    ", te_error2)

#
#n.train(tr_dt, perc.Backpropagation(10000))
#n_out = [n.predict(i) for i in inp1]
#tr_error1 = nn.RMSE(out1, n_out)
#print("Training error:", tr_error1)
#
#n_out = [n.predict(i) for i in tinp1]
#te_error1 = nn.RMSE(tout1, n_out)
#print("Test error:    ", te_error1)


