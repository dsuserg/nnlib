##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Testing perceptron model on xor problem
#
#Created on Tue Jul 31 12:42:57 2018
#
#@author: dsu
#"""
import pandas as pd
from sklearn import metrics
import copy
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import shelve

iris = load_iris()     

X, y = iris.data, iris.target

X = nn.Dataset(X)

X.tune_up()
X.normalisation_linear([0,1])
y = nn.Dataset(y[:,np.newaxis])
y.tune_up()
y.normalisation_linear([0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = nn.Dataset(X_train)

X_test = nn.Dataset(X_test)

y_train = nn.Dataset(y_train)

y_test = nn.Dataset(y_test)



n = perc.NPecrep([4, 4,4, 1], perc.sgmoidFunc)

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
    n2.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
    n_out = [n2.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n2.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    print("Степень 3*")
    n3.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
    n_out = [n3.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n3.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    print("Степень 2")
    n4.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
    n_out = [n4.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n4.predict(i) for i in X_test]
    ts_error = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    
    print("Степень 3")
    n5.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
    n_out = [n5.predict(i) for i in X_train]
    tr_error = nn.RMSE(y_train, n_out)
    print("Training error:", tr_error)
    
    n_out = [n5.predict(i) for i in X_test]
    ts_errors = nn.RMSE(y_test, n_out)
    print("Test error:", ts_error)
    
    errors_train.append(tr_error)
    errors_tst.append(ts_error)
    
    print('-'*20)






#def classer(y):
#    answ = []
#    for i in range(len(y)):
#        if y[i][0] <= 0.33:
#            answ.append(0)
#        elif y[i][0] > 0.33 and y[i][0] <= 0.66 :
#            answ.append(1)
#        elif y[i][0] > 0.66:
#            answ.append(2)
#    return answ
#
#tmp = copy.copy(y_train)
#y_train = classer(y_train)
#y_test = classer(y_test)
#
#for i in range(40):
#    print("Обычный перцетрон")
#    n.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
#    n_out = [n.predict(i) for i in X_train]
#    t = copy.copy(n_out)
#    n_out = classer(n_out)
#    tr_error = metrics.accuracy_score(y_train, n_out)
#    
#    print("Training error:", tr_error)
#    
#    n_out = [n.predict(i) for i in X_test]
#    n_out = classer(n_out)
#    ts_error = metrics.accuracy_score(y_test, n_out)
#    print("Test error:", ts_error)
#    
#    errors_train.append(tr_error)
#    errors_tst.append(ts_error)
#    
#    print("Степень 2*")
#    n4.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
#    n_out = [n4.predict(i) for i in X_train]
#    n_out = classer(n_out)
#    tr_error = metrics.accuracy_score(y_train, n_out)
#    print("Training error:", tr_error)
#    
#    n_out = [n4.predict(i) for i in X_test]
#    n_out = classer(n_out)
#    ts_error = metrics.accuracy_score(y_test, n_out)
#    print("Test error:", ts_error)
#    
#    errors_train.append(tr_error)
#    errors_tst.append(ts_error)
#    
#    print("Степень 3*")
#    n5.train(tr_dt, perc.Backpropagation_n(EPOCHES_CNT))
#    n_out = [n5.predict(i) for i in X_train]
#    n_out = classer(n_out)
#    tr_error = metrics.accuracy_score(y_train, n_out)
#    print("Training error:", tr_error)
#    
#    n_out = [n5.predict(i) for i in X_test]
#    n_out = classer(n_out)
#    ts_error = metrics.accuracy_score(y_test, n_out)
#    print("Test error:", ts_error)
#    
#    errors_train.append(tr_error)
#    errors_tst.append(ts_error)
#    
#    
#    print('-'*20)
#    
#    print("Степень 2")
#    n3.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
#    n_out = [n3.predict(i) for i in X_train]
#    n_out = classer(n_out)
#    tr_error = metrics.accuracy_score(y_train, n_out)
#    print("Training error:", tr_error)
#    
#    n_out = [n3.predict(i) for i in X_test]
#    n_out = classer(n_out)
#    ts_error = metrics.accuracy_score(y_test, n_out)
#    print("Test error:", ts_error)
#    
#    errors_train.append(tr_error)
#    errors_tst.append(ts_error)
#    
#    
#    print("Степень 3")
#    n4.train(tr_dt, perc.Backpropagation_nn(EPOCHES_CNT))
#    n_out = [n4.predict(i) for i in X_train]
#    n_out = classer(n_out)
#    tr_error = metrics.accuracy_score(y_train, n_out)
#    print("Training error:", tr_error)
#    
#    n_out = [n4.predict(i) for i in X_test]
#    n_out = classer(n_out)
#    ts_errors = metrics.accuracy_score(y_test, n_out)
#    print("Test error:", ts_error)
#    
#    errors_train.append(tr_error)
#    errors_tst.append(ts_error)

    
##with shelve.open("errors") as er:
##    er['train-iris'] = errors_train
##    er['test-iris'] = errors_tst

fig, ax = plt.subplots()
ax.set_title("Зависимость ошибки от количества эпох(Обучение)")
ax.set_xlabel("Количество эпох")
ax.set_ylabel("Ошибка обучения")


fig1, ax1 = plt.subplots()
ax1.set_title("Зависимость ошибки от количества эпох(Тест)")
ax1.set_xlabel("Количество эпох")
ax1.set_ylabel("Ошибка тестовая")


ax_x = [i for i in range(500,500*41,500)]



ax.plot(ax_x, errors_train[0::5], label = "ОРО")
ax.plot(ax_x, errors_train[1::5], label = "2*")
ax.plot(ax_x, errors_train[2::5], label = "3*")
ax.plot(ax_x, errors_train[3::5], label = "2")
ax.plot(ax_x, errors_train[4::5], label = "3")


ax1.plot(ax_x, errors_tst[0::5], label = "ОРО")
ax1.plot(ax_x, errors_tst[1::5], label = "2*")
ax1.plot(ax_x, errors_tst[2::5], label = "3*")
ax1.plot(ax_x, errors_tst[3::5], label = "2")
ax1.plot(ax_x, errors_tst[4::5], label = "3")

ax.legend()
ax1.legend()