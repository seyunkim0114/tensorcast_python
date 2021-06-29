from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from matplotlib.pyplot import figure

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
    
# def splitDataset(series,n):
#     train = series[:-n]
#     test = series[-n:]
#     return train,test

def buildModelHWES(train,test,seasonp,n,zoom):
    model = HWES(train, seasonal_periods=seasonp, trend='add',seasonal='mul')
    fit = model.fit(optimized=True,use_brute=True)
    print(fit.summary())
    forecast = fit.forecast(steps=n)
    
    ticks = range(len(train)+len(test))
    fig = plt.figure()
    past, = plt.plot(ticks[-zoom:-n], train[-zoom:-n], 'b.-', label='Traffic histroy')
    future, = plt.plot(ticks[-n:], test[-n:], 'r.-', label='Traffic future')
    predicted, = plt.plot(ticks[-n:], forecast, 'g.-', label='Traffic predicted')
    plt.legend()
    fig.show()
    plt.show()

    return forecast
