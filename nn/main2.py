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
def train_net(net, sample, epoches, n):    
    sample_x = nn.Dataset(sample[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(sample[1:,:2],  expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    sample_tr = [sample_x,sample_y]
    errors =[]
    for i in range(n):
        net.train(sample_tr, perc.Backpropagation_nn(epoches))
        n_out = [net.predict(i) for i in sample_x]
        errors.append(RMSE(sample_y, n_out))
    
   
def check_prediction(n,n2, samp):
    sample_x = nn.Dataset(samp[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(samp[1:,:2], expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
        
    n_out = [sample_x[0]]
    for i in range(1,len(sample_x)):
        n_out.append(n.predict(n_out[i-1]))
        n_out[i] = np.append(n_out[i],sample_x[i][2])
    
    n_out2 = [sample_x[0]]
    for i in range(1,len(sample_x)):
        n_out2.append(n.predict(n_out2[i-1]))
        n_out2[i] = np.append(n_out2[i],sample_x[i][2])
    
    n_out = np.array(n_out)[:,:2]
    n_out2 = np.array(n_out2)[:,:2]
    err1_0 = RMSE(sample_y, n_out)*100
    print("nn error %f" %(err1_0))
    
    err1_1 = RMSE(sample_y[:,0], n_out[:,0])*100
    err1_2 = RMSE(sample_y[:,1], n_out[:,1])*100
    

    err2_0 = RMSE(sample_y[:,:2], n_out2)*100
    print("nn2 error %f" %(err2_0))
    
    
    fig, axes = plt.subplots(2, constrained_layout=True,figsize=(10,10))
    fig.suptitle("""                 Структура {0}, Количество эпох {1} 
                    Общая ошибка сети {2}%
                    Общая ошибка сети2 {3}%""".format(n.get_structure(),
                                                        EPOCHES,
                                                          int(err1_0),
                                                          int(err2_0)))
        
    err2_1 = RMSE(sample_y[:,0], n_out2[:,0])*100
    err2_2 = RMSE(sample_y[:,1], n_out2[:,1])*100
    
    r = [sample_y.denormalisation_linear([0,1],i) for i in sample_y]
    n_out = [sample_y.denormalisation_linear([0,1],i) for i in n_out] 
    n_out2 = [sample_y.denormalisation_linear([0,1],i) for i in n_out2]
    
    
    
    time = [i*2 for i in range(0,len(samp)-1)]
    axes[0].plot(time, np.array(r)[:,0], label = "Референс",)
    axes[0].plot(time, np.array(n_out)[:,0], label = "Сеть %d" % round(err1_1) +r'%')
    axes[0].plot(time, np.array(n_out2[:])[:,0], label = "Сеть2 %d" % round(err2_1)+r'%')
    axes[0].set_xlabel("Время(ч)")
    axes[0].set_ylabel("ОП")
    axes[0].legend()
    
    axes[1].plot(time, np.array(r)[:,1], label = "Референс",)
    axes[1].plot(time, np.array(n_out)[:,1], label = "Сеть %d" % round(err1_2) +r'%')
    axes[1].plot(time, np.array(n_out2[:])[:,1], label = "Сеть2 %d" % round(err2_2)+r'%')
    axes[1].set_xlabel("Время(ч)")
    axes[1].set_ylabel("Потребление субстрата(мл)")
    axes[1].legend()
    
    
   
def check_ensemble(n,n2, samp):
    sample_x = nn.Dataset(samp[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(samp[1:,:2], expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    n_out = np.array([n.predict(i) for i in sample_x ])
    n_out2 = np.array([n2.predict(i) for i in sample_x ])
    err1_0 = RMSE(sample_y, n_out)*100
    print("nn error %f" %(err1_0))
    
    
    err1_1 = RMSE(sample_y[:,0], n_out[:,0])*100
    err1_2 = RMSE(sample_y[:,1], n_out[:,1])*100
    

    err2_0 = RMSE(sample_y, n_out2)*100
    print("nn2 error %f" %(err2_0))
    
    
    fig, axes = plt.subplots(2, constrained_layout=True,figsize=(10,10))
    fig.suptitle("""                 Структура {0}, Количество эпох {1} 
                    Общая ошибка сети {2}%
                    Общая ошибка сети2 {3}%""".format(n.get_structure(),
                                                        EPOCHES,
                                                          int(err1_0),
                                                          int(err2_0)))
        
    err2_1 = RMSE(sample_y[:,0], n_out2[:,0])*100
    err2_2 = RMSE(sample_y[:,1], n_out2[:,1])*100
    
    r = [sample_y.denormalisation_linear([0,1],i) for i in sample_y]
    n_out = [sample_y.denormalisation_linear([0,1],i) for i in n_out] 
    n_out2 = [sample_y.denormalisation_linear([0,1],i) for i in n_out2]
    
    
    
    time = [i*2 for i in range(0,len(samp)-1)]
    axes[0].plot(time, np.array(r)[:,0], label = "Референс",)
    axes[0].plot(time, np.array(n_out)[:,0], label = "Сеть %d" % round(err1_1) +r'%')
    axes[0].plot(time, np.array(n_out2[:])[:,0], label = "Сеть2 %d" % round(err2_1)+r'%')
    axes[0].set_xlabel("Время(ч)")
    axes[0].set_ylabel("ОП")
    axes[0].legend()
    
    axes[1].plot(time, np.array(r)[:,1], label = "Референс",)
    axes[1].plot(time, np.array(n_out)[:,1], label = "Сеть %d" % round(err1_2) +r'%')
    axes[1].plot(time, np.array(n_out2[:])[:,1], label = "Сеть2 %d" % round(err2_2)+r'%')
    axes[1].set_xlabel("Время(ч)")
    axes[1].set_ylabel("Потребление субстрата(мл)")
    axes[1].legend()
        
    
data = pd.read_csv("data.csv")
data = data.loc[:,["op_density","subs_consumption", "subs_flow",]]

sample_1 = data[:82]        #82
sample_2 = data[85:122]     #37
#sample_3 = data[125:146]    #21
sample_4 = data[149:206]    #57
sample_5 = data[209:269]    #60
#sample_6 = data[272:315]    #39
 

sample_1 = sample_1.to_numpy()
sample_2 = sample_2.to_numpy()
sample_4 = sample_4.to_numpy()
sample_5 = sample_5.to_numpy()


sample = np.concatenate((sample_1,sample_2,sample_4,sample_5))
#sample = sample_1
#sample = sample[:]

n = perc.NPecrep([3, 12, 2], perc.sgmoidFunc)
n2 = copy.deepcopy(n) 
n2.set_new_order(2)

EPOCHES = 100

train_net(n,sample,EPOCHES)
train_net(n2,sample,EPOCHES)


check_ensemble(n,n2, sample_1)
check_ensemble(n,n2, sample_2)
check_ensemble(n,n2, sample_4)
check_ensemble(n,n2, sample_5)


#check_prediction(n,n2, sample_1)
#check_prediction(n,n2, sample_2)
#check_prediction(n,n2, sample_4)
#check_prediction(n,n2, sample_5)



