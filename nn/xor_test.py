#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing perceptron model on xor problem

Created on Tue Jul 31 12:42:57 2018

@author: dsu
"""

import copy
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

iris = load_wine()     

X, y = iris.data, iris.target

X = nn.Dataset(X)
X.tune_up()
X.normalisation_linear([0,1])
y = nn.Dataset(y[:,np.newaxis])
y.tune_up()
y.normalisation_linear([0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train = nn.Dataset(X_train)

X_test = nn.Dataset(X_test)

y_train = nn.Dataset(y_train)

y_test = nn.Dataset(y_test)



n = perc.NPecrep([13, 13, 1], perc.sgmoidFunc)
n2 = copy.deepcopy(n)
n2.set_new_order(2,False)

n3 = copy.deepcopy(n)
n3.set_new_order(3,False)

n4 = copy.deepcopy(n2)

n5 = copy.deepcopy(n3)

tr_dt = [X_train, y_train] 
#n_out = [n2.predict(i) for i in X_train]
#tr_error1 = nn.RMSE(y_train, n_out)
#print("Training error:", tr_error1)    
EPOCHES_CNT = 500
errors_train=[]
errors_tst=[]

for i in range(40):
    print("Обычный перцетрон")
    n.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
    n_out = [n.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    print("Степень 2*")
    n4.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
    n_out = [n4.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n4.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    print("Степень 3*")
    n5.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
    n_out = [n5.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n5.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    
    print('-'*20)
    
    print("Степень 2")
    n3.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
    n_out = [n3.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n3.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    
    print("Степень 3")
    n4.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
    n_out = [n4.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n4.predict(i) for i in X_test]
    ts_errors = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)



#figtr, ax = plt.subplots()
#fig.show()
#
#figts
