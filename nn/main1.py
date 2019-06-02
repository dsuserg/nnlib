import pandas as pd
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc
import matplotlib.pyplot as plt
import copy
from numba import jit
from sklearn.metrics import mean_squared_error as MSE
import shelve
import multiprocessing as mp

def RMSE(x, y):
    return MSE(x, y)**(1/2)

def mp_train_net(child_con,*argv):
    child_con.send(train_net(*argv))
    child_con.close()
    
def train_net(net, sample, epoches, n_times):    
    sample_x = nn.Dataset(sample[:-1], expert = [[0,80], [-12,6], [0,60]])
    sample_y = nn.Dataset(sample[1:,:2],  expert = [[0,80], [-12,6]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    errors = []
    sample_tr = [sample_x,sample_y]
    
    for i in range(n_times):
        net.train(sample_tr, perc.Backpropagation_nn(epoches))
        n_out = [net.predict(i) for i in sample_x]
        errors.append(RMSE(sample_y, n_out))
    
    return errors
   
def check_error(errors_lst, epoches, n_times):
    fig, axes = plt.subplots(constrained_layout=True)
    epoche_axe = [epoches*i for i in range(1, n_times+1)]
    
    for i, errors in enumerate(errors_lst):
        axes.plot(epoche_axe, errors, label = "Сеть %s" % str(i+1))
    
    fig.suptitle("""                      Структура сети 1 {0}
                    Структура сети 2 {1}
                    Структура сети 3 {2}""".format("3-12-2",
                                                   "3-6-2",
                                                   "3-4-2"))
    axes.set_xlabel("Количество эпох")
    axes.set_ylabel("Ошибка")
    axes.legend()
#    fig.suptitle.format()
    
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
    
    

if __name__ == "__main__":    
    data = pd.read_csv("data.csv")
    data = data.loc[:,["op_density","subs_consumption", "subs_flow"]]
    #236
    sample_1 = data[:82]        #82
    sample_2 = data[85:122]     #37
    sample_3 = data[149:206]    #57
    sample_4 = data[209:269]    #60
     
    
    sample_1 = sample_1.to_numpy()
    sample_2 = sample_2.to_numpy()
    sample_3 = sample_3.to_numpy()
    sample_4 = sample_4.to_numpy()
    
    sample = np.concatenate((sample_1,sample_2,sample_3,sample_4))
    #sample = sample_1
    sample = sample[:]
    
    n1 = perc.NPecrep([3, 6, 2], perc.sgmoidFunc)
    
    n2 = perc.NPecrep([3, 3, 2], perc.sgmoidFunc)
    n2.set_new_order(2)
    
    n3 = perc.NPecrep([3, 2, 2], perc.sgmoidFunc)
    n3.set_new_order(3)
    
    
    EPOCHES = 10
    n_times = 20//EPOCHES
    
    parent1 , child1 = mp.Pipe()
    parent2 , child2 = mp.Pipe()
    parent3 , child3 = mp.Pipe()
    
    
    p1 = mp.Process(target=mp_train_net , args=(child1, n1, sample, EPOCHES, n_times))
    p2 = mp.Process(target=mp_train_net , args=(child2, n2, sample, EPOCHES, n_times))
    p3 = mp.Process(target=mp_train_net , args=(child3, n3, sample, EPOCHES, n_times))
    
    p1.start()
    p2.start()
    p3.start()
    
    err1 = parent1.recv()
    err2 = parent2.recv()
    err3 = parent3.recv()
    
    p1.join()
    p2.join()
    p3.join()
    
    
    
    
    
    with shelve.open("err1") as db:
        db["err1"] = err1
    
    with shelve.open("err2") as db:
        db["err2"] = err2
    
    
    with shelve.open("err3") as db:
        db["err3"] = err3
    
    
    print("done")
#    check_error((err1,err2,err3), EPOCHES, n_times)
        
    
    
    #check_ensemble(n,n2, sample_1)
    #check_ensemble(n,n2, sample_2)
    #check_ensemble(n,n2, sample_4)
    #check_ensemble(n,n2, sample_5)
    
    
    #with shelve.open("err1") as db:
    #    err1 = db["err1"] 
    #
    #with shelve.open("err2") as db:
    #    err2 = db["err2"]  
    #
    #with shelve.open("err3") as db:
    #    err3 = db["err3"] 
    #
    #check_error((err1,err2,err3), EPOCHES, n_times)
    
