import pandas as pd
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy
from numba import jit
from sklearn.metrics import mean_squared_error as MSE

def RMSE(x, y):
    return MSE(x, y)**(1/2)

@jit(cache=True)
def train_net(net, sample, epoches):    
    sample_x = nn.Dataset(sample[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(sample[1:,:2],  expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    sample_tr = [sample_x,sample_y]
    net.train(sample_tr, perc.Backpropagation_nn(epoches))
    n_out = [net.predict(i) for i in sample_x]
    
    print(RMSE(sample_y, n_out))


def ensemble_predict(ensemble,example):
    outs = [net.predict(example) for net in ensemble]
    return(sum(outs)/len(outs))
    

def check_ensemble(ensemble, samp):
    sample_x = nn.Dataset(samp[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(samp[1:,:2], expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    n_out = np.array([ensemble_predict(ensemble,i) for i in sample_x ])
    
    err1_0 = RMSE(sample_y, n_out)*100
    print("nn error %f" %(err1_0))
    
    err1_1 = RMSE(sample_y[:,0], n_out[:,0])*100
    err1_2 = RMSE(sample_y[:,1], n_out[:,1])*100
#    err1_3 = RMSE(sample_y[:,2], n_out[:,2])*100
    
#    prediction = [sample_x[0]]
#    for i in range(1,len(sample_x)):
#        prediction.append(ensemble_predict(ensemble, prediction[i-1]))
#        prediction[i] = np.append(prediction[i],sample_x[i][3])
#    
#    prediction = np.array(prediction)
#    err2_0 = RMSE(sample_y, prediction[:,:3])*100
#    print("ensemble prediction error %f" %(err2_0))
    
    
    fig, axes = plt.subplots(2, constrained_layout=True,figsize=(10,10))
    fig.suptitle("""                 Структура {0}, Количество эпох {1} 
                    Общая ошибка {2}%
                    """.format(ensemble[0].get_structure(),
                               EPOCHES,
                               int(err1_0)
                               ))
        
#    err2_1 = RMSE(sample_y[:,0], prediction[:,0])*100
#    err2_2 = RMSE(sample_y[:,1], prediction[:,1])*100
#    err2_3 = RMSE(sample_y[:,2], prediction[:,2])*100
    
    r = [sample_y.denormalisation_linear([0,1],i) for i in sample_y]
    n_out = [sample_y.denormalisation_linear([0,1],i) for i in n_out] 
#    prediction = [sample_y.denormalisation_linear([0,1],i) for i in prediction]
    
    
    
    
    time = [i*2 for i in range(0,len(samp)-1)]
    axes[0].plot(time, np.array(r)[:,0], label = "Референс",)
    axes[0].plot(time, np.array(n_out)[:,0], label = "Отклик %d" % round(err1_1) +r'%')
#    axes[0].plot(time, np.array(prediction[:])[:,0], label = " Прогноз %d" % round(err2_1)+r'%')
    axes[0].set_xlabel("Время(ч)")
    axes[0].set_ylabel("ОП")
    axes[0].legend()
    
    axes[1].plot(time, np.array(r)[:,1], label = "Референс",)
    axes[1].plot(time, np.array(n_out)[:,1], label = "Отклик %d" % round(err1_2) +r'%')
#    axes[1].plot(time, np.array(prediction[:])[:,1], label = "Прогноз %d" % round(err2_2)+r'%')
    axes[1].set_xlabel("Время(ч)")
    axes[1].set_ylabel("Потребление субстрата(мл)")
    axes[1].legend()
    
    
#    axes[2].plot(time, np.array(r)[:,2], label = "Референс",)
#    axes[2].plot(time, np.array(n_out)[:,2], label = "Отклик %d" % round(err1_3) +r'%')
##    axes[2].plot(time, np.array(prediction[:])[:,2], label = "Прогноз %d" % round(err2_3)+r'%')
#    axes[2].set_xlabel("Время(ч)")
#    axes[2].set_ylabel("Каратин(%)")
#    axes[2].legend()
    
    
data = pd.read_csv("data.csv")
data = data.loc[:,["op_density","subs_consumption", "subs_flow",]]

#236
sample_1 = data[:82]        #82
sample_2 = data[85:122]     #37
#sample_3 = data[125:146]    #21
sample_4 = data[149:206]    #57
sample_5 = data[209:269]    #60
#sample_6 = data[272:315]    #39
 
sample_1 = sample_1.to_numpy()
sample_2 = sample_2.to_numpy()
#sample_3 = sample_3.to_numpy()
sample_4 = sample_4.to_numpy()
sample_5 = sample_5.to_numpy()
#sample_6 = sample_6.to_numpy()

n1 = perc.NPecrep([3, 3, 2], perc.sgmoidFunc)
#n1.set_new_order(2)
n2 = copy.deepcopy(n1)
n3 = copy.deepcopy(n1)
n4 = copy.deepcopy(n1)
n5 = copy.deepcopy(n1)
n6 = copy.deepcopy(n1)


EPOCHES = 1000

train_net(n1,sample_1,EPOCHES)
train_net(n2,sample_2,EPOCHES)
#train_net(n3,sample_3,EPOCHES)
train_net(n4,sample_4,EPOCHES)
train_net(n5,sample_5,EPOCHES)
#train_net(n6,sample_6,EPOCHES)

ensemble = [n1,n2,n4,n5]
#ensemble = [n1]
#ensemble1 = [n1]
#ensemble2 = [n2]
#ensemble3 = [n3]
#ensemble4 = [n4]
#ensemble5 = [n5]
#ensemble6 = [n6]


check_ensemble(ensemble, sample_1)
check_ensemble(ensemble, sample_2)
#check_ensemble(ensemble, sample_3)
check_ensemble(ensemble, sample_4)
check_ensemble(ensemble, sample_5)
#check_ensemble(ensemble, sample_6)





