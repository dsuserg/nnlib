import pandas as pd
import numpy as np
import nnlibs.nnlib as nn
import nnlibs.perceptron as perc
import matplotlib.pyplot as plt
import multiprocessing as mp

def train_net(net, sample, epoches):    
    sample_x = nn.Dataset(sample[:-1], expert = [[0,80], [-12,6], [0,4], [0,60]])
    sample_y = nn.Dataset(sample[1:,:3],  expert = [[0,80], [-12,6], [0,4]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    sample_tr = [sample_x,sample_y]
    net.train(sample_tr, perc.Backpropagation_nn(epoches))
    n_out = [net.predict(i) for i in sample_x ]
    
#    print("Done")
    print( nn.RMSE(sample_y, n_out))

def ensemble_predict(ensemble,example):
    outs = [net.predict(example) for net in ensemble]
    return(sum(outs)/len(outs))
    
def check_ensemble(samp):
    sample_x = nn.Dataset(samp[:-1], expert = [[0,80], [-12,6],[0,4], [0,60]])
    sample_y = nn.Dataset(samp[1:,:3], expert = [[0,80], [-12,6],[0,4]])
    sample_x.normalisation_linear([0,1])
    sample_y.normalisation_linear([0,1])
    
    n_out = np.array([ensemble_predict(ensemble,i) for i in sample_x ])
    
    err = nn.RMSE(sample_y, n_out)
    print("error %f" %(err))
    
    fig, ax = plt.subplots()
    ax.plot(range(0,len(samp)-1), sample_y[:,0])
    ax.plot(range(0,len(samp)-1), n_out[:,0])
    
    #prediction = [sample_x[0]]
    #for i in range(len(sample_x)):
    #    prediction.append(ensemble_predict(ensemble,prediction[i-1]))
    #
    #plt.plot(range(0,len(sample_x)), np.array(prediction[1:])[:,0])
    
    prediction = [sample_x[0]]
    for i in range(1,len(sample_x)):
        prediction.append(ensemble_predict(ensemble, prediction[i-1]))
        prediction[i] = np.append(prediction[i],sample_x[i][3])
    ax.plot(range(0,len(sample_x)), np.array(prediction[:])[:,0])
    
    
    

data = pd.read_csv("data.csv")
data = data.loc[:,["op_density","subs_consumption", "caratin", "subs_flow",]]

sample_1 = data[:82]        #82
sample_2 = data[85:122]     #37
sample_3 = data[125:146]    #21
sample_4 = data[149:206]    #57
sample_5 = data[209:269]    #60
sample_6 = data[272:315]    #39
 
sample_1 = sample_1.to_numpy()
sample_2 = sample_2.to_numpy()
sample_3 = sample_3.to_numpy()
sample_4 = sample_4.to_numpy()
sample_5 = sample_5.to_numpy()
sample_6 = sample_6.to_numpy()

n1 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)
n2 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)
n3 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)
n4 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)
n5 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)
n6 = perc.NPecrep([4, 4,4, 3], perc.sgmoidFunc)


EPOCHES = 100000


p1 = mp.Process(target = train_net, args = (n1, sample_1, int(EPOCHES),))
p2 = mp.Process(target = train_net, args = (n2, sample_2, int(EPOCHES),))
p3 = mp.Process(target = train_net, args = (n3, sample_3, int(EPOCHES),))
p4 = mp.Process(target = train_net, args = (n4, sample_4, int(EPOCHES),))
p5 = mp.Process(target = train_net, args = (n5, sample_5, int(EPOCHES),))
p6 = mp.Process(target = train_net, args = (n6, sample_6, int(EPOCHES),))

p1.start()
p1.join()
p2.start()
p2.join()
p3.start()
p3.join()
p4.start()
p4.join()
p5.start()
p5.join()
p6.start()
p6.join()



#print(err1)
#print(err2)
#print(err3)
#print(err4)
#print(err5)
#print(err6)


ensemble = [n1,n2,n3,n4,n5,n6]

check_ensemble(sample_1)
check_ensemble(sample_2)
check_ensemble(sample_3)
check_ensemble(sample_4)
check_ensemble(sample_5)
check_ensemble(sample_6)





