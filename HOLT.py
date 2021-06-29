from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from matplotlib.pyplot import figure
import numpy as np

time_list = [0,1,2,3,4]

def getDataset(csv_file):
    T = pd.read_csv(csv_file, index_col=False)
    return T

def printFigure(fac,w,h):
    num = len(time_list)
    plt.figure()
    figure(figsize=(w,h), dpi=80)
    for i in range(0,num):
        plt.subplot(1,num,i+1)
        fac[str(time_list[i])].plot()
        plt.title(f'{time_list[i]}th')
        


def buildModelHolt(train,test,n,zoom):
    model = Holt(train,exponential=True)
    fit = model.fit(optimized=True,use_brute=True)
    print(fit.summary())
    forecast = fit.forecast(steps=n)
    print(f'Forecast for next {n} time steps : {forecast}')

    ticks = range(len(train)+len(test))
    fig = plt.figure()
    past, = plt.plot(ticks[-zoom:-n], train[-zoom:-n], 'b.-', label='Traffic histroy')
    future, = plt.plot(ticks[-n:], test[-n:], 'r.-', label='Traffic future')
    predicted, = plt.plot(ticks[-n:], forecast, 'g.-', label='Traffic predicted')
    plt.legend()
    fig.show()
    plt.show()
    
    return forecast
