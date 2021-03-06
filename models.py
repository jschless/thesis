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
from sklearn.linear_model import RidgeClassifier
from models import *

class TimeLag:
    def __init__(self, n_days):
        self.n = n_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Takes X dataframe and lags it by self.n days"""
        X = X.to_frame()
        temp = X
        for i in range(1,self.n+1):
            temp2 = temp.shift(i).rename(columns={'Close': 'Close' + str(i)})
            X = pd.merge(X, temp2, on='Date')
        X = X[X.columns.drop('Close')]
        return X.iloc[self.n:]
    
class Stock:
    def __init__(self, name, timePeriod, train_length = -1, debug=False):
        """Initializes a stock object
        
        Keyword arguments:
        name - stock symbol
        timePeriod - tuple (start, end) of period of interest in datetime
        train_length - optional argument to limit the length of training
        """
        self.name = name
        self.timePeriod = timePeriod
        self.train_length = train_length
        self.data, self.trainData, self.testData = self.getData()
        self.startDate = timePeriod[0]
        self.endDate = timePeriod[1]
        self.n_days_test = len(self.testData) 
        self.debug = debug      

    def getYield(self):
        stockStart = self.testData['Open'].iloc[0]
        stockEnd = self.testData['Close'].iloc[-1]
        stockYield = (stockEnd-stockStart)/stockStart
        return stockYield*100
        
    def __str__(self):
        return self.name + " from " + str(self.startDate) + " to " + str(self.endDate)

    def getData(self, cats=['Close', 'Open']):
        """Reads stock data from CSV"""
        path = "C:\\Users\\x92423\Documents\\Thesis Data Grab\\" + str(self.name) + ".csv" 
        series = pd.read_csv(path, parse_dates=[0], index_col=0)
        columnsToDrop = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        for cat in cats:
            columnsToDrop.remove(cat)
        series = series.drop(columns=columnsToDrop).dropna()
        startTest = 0
        startTrain = 0
        if self.train_length > 0:
            series = series[self.timePeriod[0]-datetime.timedelta(days=self.train_length):self.timePeriod[1]]
        classSeries = [1 if series[cat][i] > series[cat][i-1] else 0 for i in range(len(series))]            
        series['Classification'] = classSeries
        test_series = series.loc[self.timePeriod[0]:]
        train_series = series[:self.timePeriod[0]]
        return  series, train_series, test_series

    def getDayPriceClose(self, i):
        return self.testData.iloc[i]['Close']

    def getDayPriceOpen(self, i):
        return self.testData.iloc[i]['Open']


class Model:
    def __init__(self, stock, params={'lag':5}, param_ranges={'lag':range(2,20,2)}, debug=False):
        """Initializes model"""
        self.mod = LinearRegression()   
        self.name = 'LINREG'
        self.params = params
        self.param_ranges = param_ranges
        self.debug = debug
        self.investments = {}
        self.performance = {}
        self.stock=stock
        self.yields = {}
        self.predictedYs = []
        self.actualYs = []
        self.pYields = []
        self.cashStock = {}
        self.classification = False
         
    def __str__(self):
        return "Linear Regression Model"
        
    def addPerformance(self, alpha, performance):
        self.performance[alpha] = performance

    def addCashStock(self, alpha, cashStock):
        self.cashStock[alpha] = cashStock
        
    def addInvestments(self, alpha, investments):
        self.investments[alpha] = investments

    def addYield(self, alpha, pyield):
        self.yields[alpha] = pyield
            
    def fit(self, X, y):
        self.mod.fit(X, y)

    def score(self, X, y):
        return self.mod.score(X,y)
        
    def initMod(self, data, params):
        self.params = params
        self.lag_n = params['lag']
        self.lag = TimeLag(self.lag_n)
        self.laggedData = self.lag.transform(data)

    def validate(self, day, n_splits = 2, kfold = True):        
        kf = KFold(n_splits=2)
        dayBefore= day-datetime.timedelta(days=1)

        combinations = self.generateCombinations(self.param_ranges)
        bestParams = []
        bestScore = -100
        X = self.stock.data['Close'][:day]
        cat = 'Classification' if self.classification else 'Close'
        if self.debug:
            print("input for model " + str(X.tail()))
        if not kfold:
            self.initMod(X, self.params)
            y = self.stock.data[cat][:dayBefore].iloc[self.lag_n:]
            self.fit(self.laggedData[:dayBefore], y)
            return
        for combo in combinations:
            total = 0
            self.initMod(X, combo)
            y = self.stock.data[cat][:dayBefore].iloc[self.lag_n:]
            for train_index, test_index in kf.split(self.laggedData[:dayBefore]):                
                X_train, X_test = self.laggedData.iloc[train_index], self.laggedData.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.fit(X_train, y_train)
                total += self.score(X_test, y_test)
            if self.debug:
                print("total score: " + str(total) + "   for params: " + str(combo) + "   avg score: " + str(total/n_splits))
            if total/n_splits > bestScore:
                bestScore = total/n_splits
                bestParams = combo
        if self.debug:
            print("model validated, chosing params: " + str(bestParams))
        self.initMod(X, bestParams)
        y = self.stock.data[cat][:dayBefore].iloc[self.lag_n:]
        self.fit(self.laggedData[:dayBefore], y)

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
        return comboDicts

    def getYields(self, validationFreq=0):
        pYields = []
        validationDays = self.numValidations(validationFreq)
        predictedYs = []
        actualYs = []
        for i in range(len(self.stock.testData)):
            day = self.stock.testData.index[i]
            self.validate(day, kfold= (i in validationDays))
            if self.debug:
                print("training model for day " + str(day))
                print("Lagged data for day " + str(day) + " : " + str(self.laggedData[day:day]))
            predictY = self.mod.predict(self.laggedData[day:day])
            oldY = self.stock.testData.iloc[i]['Open']
            actualY = self.stock.testData.iloc[i]['Close']
            pYield = (predictY-oldY)/oldY
            
            if self.classification:
                conf = self.mod.decision_function(self.laggedData[day:day])
                pYield = conf
                predictY = [actualY + 5] if predictY == 1 else [actualY-5]
            pYields.append(pYield[0])
#            if self.name=='LASSO' or self.name=='RIDGE' or self.name=='RIDGECLASS' or self.name=='MLP':
            predictedYs.append(predictY[0])
#            else:
#                print (self.name)
#                predictedYs.append(predictY[0][0])
            actualYs.append(actualY)
        self.predictedYs = predictedYs
        self.actualYs = actualYs
        self.pYields = pYields
        self.meanError = sum(map(lambda x,y: abs(x-y), predictedYs, actualYs))/ len(predictedYs)
        return pYields
    
class LassoModel(Model):
    def __init__(self, stock, params = {'alpha': 1.0, 'lag':5}, param_ranges = {'alpha': np.logspace(-2,1,num=4), 'lag':range(2,10,4)}, debug=False):
        super(LassoModel, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.a = params['alpha']
        self.mod = Lasso(self.a)
        self.name = 'LASSO'

    def initMod(self, data, params):
        self.lag_n = params['lag']
        self.lag = TimeLag(self.lag_n)
        self.laggedData = self.lag.transform(data)
        self.a = params['alpha']
        self.mod = Lasso(self.a)
        
    def __str__(self):
        return "Lasso Regression Model"

class RidgeModel(Model):
    def __init__(self, stock, params = {'alpha': 1.0, 'lag':5}, param_ranges = {'alpha': np.logspace(-2,1,num=4), 'lag':range(2,10,2)}, debug=False):
        super(RidgeModel, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.a = params['alpha']
        self.mod = Ridge(self.a)
        self.name = 'RIDGE'
    def __str__(self):
        return "Ridge Regression Model"
    
class DCA(Model):
    def __init__(self,stock, interval=5, debug=False):
        super(DCA, self).__init__(stock, debug=debug)
        self.name = 'DCA'
        self.interval = interval
    def __str__(self):
        return "Dollar Cost Averaging"
    def fit(self, X, y):
        return self #no need to fit
    def getYields(self, validationFreq=0):
        self.meanError = 0
        return [1 if i%self.interval == 0 else 0 for i in range(self.stock.n_days_test)]

class ARIMAModel(Model):
    def __init__(self, stock, n=1, p=0, q=0, params = {}, param_ranges = {}, debug=False):
        super(ARIMAModel, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.n=n
        self.p=p
        self.q=q
        self.name='ARIMA'
    def __str__(self):
        return "ARIMA"

    def getYields(self, validationFreq = 0):
        history = [x for x in self.stock.trainData]
        pYields = []
        for i in range(len(self.stock.testData)):
            self.mod = ARIMA(history, order=(self.p,self.p,self.q))
            model_fit = self.mod.fit(disp=-1)
            output = model_fit.forecast()
            predictY = output[0]
            actualY = self.stock.testData.iloc[i]['Close']
            pYield = (predictY-actualY)/actualY
            pYields.append(pYield)
            history.append(actualY)
        return pYields

class MLP(Model):
    def __init__(self, stock, params = {'hidden layers': (100,), 'alpha':.0001, 'lag': 5}, param_ranges = {'alpha': np.logspace(-5,1, num=4), 'lag' : range(2,10,3), 'hidden layers': [(100,), (1000,), (500,)]}, debug=False):
        super(MLP, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.a = params['alpha']
        self.hidden_layer = params['hidden layers']
        self.lag = TimeLag(params['lag'])
        self.mod = MLPRegressor(hidden_layer_sizes = self.hidden_layer, alpha=self.a, max_iter=500)
        self.name = 'MLP'
        
    def __str__(self):
        return "Multi-Layer Perceptron Model"

    def fit(self, X, y):
        self.mod.fit(X, np.ravel(y))
'''
class RNN(Model):
    def __init__(self, stock, params = {'hidden layers': (10,), 'alpha':.0001, 'lag': 5}, param_ranges = {'alpha': np.logspace(-5,1, num=4), 'lag' : range(2,10,3)}, debug=False):
        super(RNN, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.a = params['alpha']
        self.hidden_layer = params['hidden layers']
        self.lag = TimeLag(params['lag'])
        self.mod = MLPRegressor(hidden_layer_sizes = self.hidden_layer, alpha=self.a)
        self.name = 'RNN'
        
    def __str__(self):
        return "Recurrent Neural Network Model"

    def fit(self, X, y):
        self.mod.fit(X, np.ravel(y))
'''
        
class RidgeClass(Model):
    def __init__(self, stock, params = {'alpha': 1, 'lag':2}, param_ranges = {'alpha': np.logspace(-5,1, num=4), 'lag' : range(2,10,3)}, debug=False):
        super(RidgeClass, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.a = params['alpha']
        self.lag = TimeLag(params['lag'])
        self.mod = RidgeClassifier(alpha=self.a)
        self.name = 'RIDGECLASS'
        self.classification = True

    def __str__(self):
        return "Ridge Classifier"
