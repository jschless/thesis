from statsmodels.tsa.arima_model import ARIMA
import csv
import sys
import time
import datetime
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from models import *
from thinkAndGrowRich import *
def tester():
    stocks = ['GOOG']
    models = ['LASSO', 'DCA']
    timeFrame = (datetime.date(2015,6,20), datetime.date(2015,8,20))
    principal = 35000 #amount of money starting the investment with
    zeroGainTimeFrame = (datetime.date(2013 ,12, 2), datetime.date(2014, 5, 12))
    #$531.48 -> $517.78
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = np.linspace(1,4,4)
    test = Simulation(stocks, models, zeroGainTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    #test.plotPerformanceByStock()
    #test.alphaPlot()
    test.plotInvestmentAmounts()
def marketCrash():
    stocks = ['GOOG']
    models = ['RIDGE', 'DCA']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2008,11,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = [1]#,2]#np.linspace(1,4,2)
    test = Simulation(stocks, models, negativeTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.plotGenerator()

def debug():
    stocks = ['GOOG']
    models = ['RIDGE']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2007,12,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = [1]#,2]#np.linspace(1,4,2)
    test = Simulation(stocks, models, negativeTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=True, alphas=alphas)
    test.run()
    test.plotGenerator()

def alphaTester():
    stocks = ['GOOG']
    models = ['RIDGE', 'DCA']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2008,11,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = np.linspace(1,4,6)
    test = Simulation(stocks, models, negativeTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.alphaPlot()

def simpleTest():
    stock = 'GOOG'
    models = ['LASSO', 'DCA']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2008,11,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = [10]
    zeroGainTimeFrame = (datetime.date(2013 ,12, 2), datetime.date(2014, 5, 12))
    test = Simulation(stock, models, zeroGainTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.plotStuff()
simpleTest()

    
#alphaTester()
