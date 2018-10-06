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


class TimeLag:
    def __init__(self, n_days):
        self.n = n_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        temp = X
        for i in range(1,self.n+1):
            temp2 = temp.shift(i).rename(columns={'Close': 'Close' + str(i)})
            X = pd.merge(X, temp2, on='Date')
        X = X[X.columns.drop('Close')]
        return X.iloc[self.n:]

class Stock:
    def __init__(self, name, timePeriod, train_length = -1):
        self.name = name
        self.timePeriod = timePeriod
        self.train_length = train_length
        self.closeData, self.closeTrainData, self.closeTestData = self.getData('Close')
        self.openData, self.openTrainData, self.openTestData = self.getData('Open')
        self.startDate = timePeriod[0]
        self.endDate = timePeriod[1]
        self.n_days_test = len(self.closeTestData) 
        

    def __str__(self):
        return self.name + " from " + str(self.startDate) + " to " + str(self.endDate)

    def getData(self, cat='Close'):
        path = "C:\\Users\\x92423\Documents\\Thesis Data Grab\\" + str(self.name) + ".csv" 
        series = pd.read_csv(path, parse_dates=[0], index_col=0)
        columnsToDrop = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        columnsToDrop.remove(cat)
        series = series.drop(columns=columnsToDrop).dropna()
        startTest = 0
        startTrain = 0
        if self.train_length > 0:
            series = series[self.timePeriod[0]-datetime.timedelta(days=self.train_length):self.timePeriod[1]]
        test_series = series.loc[self.timePeriod[0]:]
        train_series = series[:self.timePeriod[0]]
        return  series, train_series, test_series

    def getDayPriceClose(self, i):
        return self.closeTestData.iloc[i]['Close']


class Model:
    def __init__(self, params={'lag':5}, param_ranges={'lag':range(2,20,2)}):
        self.mod = LinearRegression()   
        self.name = 'LINREG'
        self.params = params
        self.param_ranges = param_ranges

    def __str__(self):
        return "Linear Regression Model"
        
    def addStock(self, stock):
        self.stock = stock

    def fit(self, X, y):
        self.mod.fit(X, y)

    def score(self, X, y):
        return self.mod.score(X,y)
        

    def initMod(self, data, params):
        self.params = params
        self.lag_n = params['lag']
        self.lag = TimeLag(self.lag_n)
        self.laggedData = self.lag.transform(data)
       
    '''
    inputs:
    day : what day are we looking at
    n_splits: k in k-fold
    output:
    model fit according to best parameters
    '''
    def validate(self, day, n_splits = 2, kfold = True):        
        kf = KFold(n_splits=2)
        combinations = self.generateCombinations(self.param_ranges)
        bestParams = []
        bestScore = -100
        X = self.stock.closeData[:day]
        if not kfold:
            self.initMod(X, self.params)
            self.fit(self.laggedData, X.iloc[self.lag_n:])
            return
        for combo in combinations:
            total = 0
            self.initMod(X, combo)
            y = self.stock.closeData[:day].iloc[self.lag_n:]
            for train_index, test_index in kf.split(self.laggedData):                
                X_train, X_test = self.laggedData.iloc[train_index], self.laggedData.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.fit(X_train, y_train)
                total += self.score(X_test, y_test)
            #print("total score: " + str(total) + "   for params: " + str(combo))
            if total/n_splits > bestScore:
                bestScore = total/n_splits
                bestParams = combo
        print("model validated, chosing params: " + str(bestParams))
        self.initMod(X, bestParams)
        self.fit(self.laggedData, X.iloc[self.lag_n:])

    '''
    performs validations every freq days
    
    '''
    def numValidations(self, freq):
        if freq == 0:
            return [0]
        else:
            return range(0, self.stock.n_days_test, freq)
        
    def generateCombinations(self, params):
        options = []
        keys = []
        for key, value in params.items():
            options.append(value)
            keys.append(key)
        combos =  [x for x in itertools.product(*options)]        
        comboDicts = []
        for combo in combos:
            temp = {}
            for i in range(len(keys)):
                temp[keys[i]] = combo[i]
            comboDicts.append(temp)
        #print('potential param options: ' + str(comboDicts))
        return comboDicts

    '''
    inputs:
    kfolds: number of validations to do
    '''
    def getYields(self, validationFreq=0):
        pYields = []
        #self.validate(self.stock.testData.index[0]) #validate off of the first day
        validationDays = self.numValidations(validationFreq)
        for i in range(len(self.stock.closeTestData)):
            day = self.stock.closeTestData.index[i]
            if i in validationDays:
                self.validate(day, kfold=True)
            else:
                self.validate(day, kfold=False)
            predictY = self.mod.predict(self.laggedData[day:day])
            actualY = self.stock.closeTestData.iloc[i]['Close']
            pYield = (predictY-actualY)/actualY
            pYields.append(pYield[0])
        return pYields

class LassoModel(Model):
    def __init__(self, params = {'alpha': 1.0, 'lag':5}, param_ranges = {'alpha': np.logspace(-2,1,num=4), 'lag':range(2,10,4)}):
        self.a = params['alpha']
        self.mod = Lasso(self.a)
        self.name = 'LASSO'
        self.params = params
        self.param_ranges = param_ranges

    def initMod(self, data, params):
        self.lag_n = params['lag']
        self.lag = TimeLag(self.lag_n)
        self.laggedData = self.lag.transform(data)
        self.a = params['alpha']
        self.mod = Lasso(self.a)
        
    def __str__(self):
        return "Lasso Regression Model"

class RidgeModel(Model):
    def __init__(self, params = {'alpha': 1.0, 'lag':5}, param_ranges = {'alpha': np.logspace(-2,1,num=4), 'lag':range(2,10,2)}):
        self.a = params['alpha']
        self.mod = Ridge(self.a)
        self.params = params
        self.param_ranges = param_ranges
        self.name = 'RIDGE'

    def __str__(self):
        return "Ridge Regression Model"

class DCA(Model):
    def __init__(self, interval=5):
        self.name = 'DCA'
        self.interval = interval
    def __str__(self):
        return "Dollar Cost Averaging"
    def fit(self, X, y):
        return self #no need to fit
    def getYields(self, validationFreq=0):
        pYields = []
        total = 0
        for i in range(self.stock.n_days_test):
            if (i%self.interval == 0):
                pYields.append(1)
                total += 1
            else:
                pYields.append(0)
        return list(map(lambda a : a/total, pYields))

class ARIMAModel(Model):
    def __init__(self, n=1, p=0, q=0, params = {}, param_ranges = {}):
        self.n=n
        self.p=p
        self.q=q
        self.name='ARIMA'
        self.params = params
        self.param_ranges = param_ranges

    def __str__(self):
        return "ARIMA"

    def getYields(self, validationFreq = 0):
        history = [x for x in self.stock.closeTrainData]
        pYields = []
        for i in range(len(self.stock.closeTestData)):
            self.mod = ARIMA(history, order=(self.p,self.p,self.q))
            model_fit = self.mod.fit(disp=-1)
            output = model_fit.forecast()
            predictY = output[0]
            actualY = self.stock.closeTestData.iloc[i]['Close']
            pYield = (predictY-actualY)/actualY
            pYields.append(pYield)
            history.append(actualY)
        return pYields

class MLP(Model):
    def __init__(self, params = {'hidden layers': (10,), 'alpha':.0001, 'lag': 5}, param_ranges = {'alpha': np.logspace(-5,1, num=4), 'lag' : range(2,10,3)}):
        self.a = params['alpha']
        self.hidden_layer = params['hidden layers']
        self.lag = TimeLag(params['lag'])
        self.mod = MLPRegressor(hidden_layer_sizes = self.hidden_layer, alpha=self.a)
        self.name = 'MLP'
        self.params = params
        self.param_ranges = param_ranges

    def __str__(self):
        return "Multi-Layer Perceptron Model"

    def fit(self, X, y):
        self.mod.fit(X, np.ravel(y))
