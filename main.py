import tensorly as tl
import numpy as np
from HWES import *
from tensorcast import *
from HOLT import *
from statsmodels.tsa.holtwinters import Holt

def generateSynthetic(N,M,K,R):
    A = np.random.rand(N,R)
    B = np.random.rand(M,R)
    C = np.ones((K,R))
    for i in range(R):
        np.random.seed()
        seq = np.asarray(range(0,K))
#         trend = seq*2
#         sin = np.sin(2*np.pi)
#         noise = np.random.normal(0,1,K)
        col = seq*2 * (np.sin(seq)+1) + np.random.uniform(0,1,K) 
#         col = col/max(col)
        C[:,i] = C[:,i]*col
    A = tl.tensor(A)
    B = tl.tensor(B)
    C = tl.tensor(C)
    return A,B,C,col

def getDiff(A,B,C,Ap,Bp,Cp):
    a = tl.sum((A-Ap)**2)
    b = tl.sum((B-Bp)**2)
    c = tl.sum((C-Cp)**2)
    print(f'Non temporal factors difference:\nA: {a},\tB: {b}')
    print(f'Temporal factors difference:\nC: {c}')
    return a,b,c

def splitDataset(series,n):
    train = series[:-n]
    test = series[-n:]
    return train,test

# hyperparameters
N = 30
M = 30
K = 100
r = 5
d = 3
maxiters = 50
alpha = 0
split=5
zoom=20
period = int(2*np.pi)
A,B,C,col = generateSynthetic(N,M,K,r)

# plt.plot(col)

A_pred,B_pred,T_pred = CoupledTensorFac(A,B,C,N,M,K,r,d,alpha,maxiters,False)

saveFac(T_pred,'synthetic')
Tsynthetic = np.load('Tsynthetic.npy')
T_train, T_test = splitDataset(Tsynthetic[:,0],split)
holt_forecast = buildModelHolt(T_train,T_test,split,zoom)

hw_forecast = buildModelHWES(T_train, T_test, period,split,zoom)
