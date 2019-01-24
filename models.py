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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
    
class Stock:
    def __init__(self, name, timePeriod, train_length = -1, difference=1, n_timelag=4, otherStocks=None, debug=False):
        """Initializes a stock object
        
        Keyword arguments:
        name - stock symbol
        timePeriod - tuple (start, end) of period of interest in datetime
        train_length - optional argument to limit the length of training
        """
        self.name = name
        self.startDate, self.endDate = timePeriod
        self.train_length = train_length
        self.n_difference = difference
        self.n_timelag = n_timelag
        self.otherStocks = otherStocks
        self.getData(self.otherStocks)
        self.n_days_test = len(self.testData) 
        self.debug = debug

    def __str__(self):
        return self.name + " from " + str(self.startDate) + " to " + str(self.endDate)

    def resetData(self, n_timelag):
        self.n_timelag = n_timelag
        self.getData(self.otherStocks)
        
    def getYield(self):
        stockStart = self.testData['Open'].iloc[0]
        stockEnd = self.testData['Close'].iloc[-1]
        stockYield = (stockEnd-stockStart)/stockStart
        return stockYield*100

    def classify(self, pyield):
        classes = [-3,-.5,.5,3]
        for i in range(len(classes)):
            if pyield < classes[i]:
                return i
        return len(classes)

    def declassify(self, classification):
        mapClass = {0: -3, 1: -.5, 2: 0, 3: .5, 4: 3}
        return mapClass[classification]

    def timeLag(self, X, cat, n=4):
        for i in range(1,n+1):
            X[cat + str(i)] = X[cat].shift(i)
        X = X.drop(columns=[cat])
        return X
    
    def getData(self, otherStocks, cats=['Close', 'Open', 'Volume', 'High', 'Low']):
        """Reads stock data from CSV"""
        path = "C:\\Users\\x92423\Documents\\Thesis Data Grab\\" + str(self.name) + ".csv"
        series = pd.read_csv(path, parse_dates=[0], index_col=0)
        series = series[cats].dropna()    
    
        ### ADDING OTHER STOCKS ###
        if otherStocks is not None:
            for s in otherStocks:
                path2 = "C:\\Users\\x92423\Documents\\Thesis Data Grab\\" + s + ".csv"
                temp = pd.read_csv(path2, parse_dates=[0], index_col=0)
                temp = temp[cats].dropna()
                for cat in cats:
                    series[s + ' ' + cat] = temp[cat]

        classSeries = [self.classify(100*(series['Close'][i] - series['Close'][i-1])/series['Close'][i-1]) for i in range(len(series))]
        series['Classification'] = classSeries

        self.data = series.iloc[self.n_difference+self.n_timelag:]
        self.testData =  self.data.loc[self.startDate: self.endDate]
        self.trainData =  self.data[: self.startDate]

        ### DIFFERENCING ###        
        differencedData =  self.trendDifferencing(series,  self.n_difference).dropna()
        
        ### LAGGING VALUES ###
        for cat in cats:
            differencedData = self.timeLag(differencedData, cat, self.n_timelag)

        if otherStocks is not None:
            for s in otherStocks:
                for cat in cats:
                    differencedData = self.timeLag(differencedData, s+ ' ' + cat, self.n_timelag)
    
        self.differencedData = differencedData[self.n_timelag:].drop(columns=['Classification'])
        #don't want people to accidentally use classification
    
        self.differencedTestData =  self.differencedData.loc[self.startDate:self.endDate]
        self.differencedTrainData =  self.differencedData[: self.startDate]
    
    def getDayPriceClose(self, i):
        return self.testData.iloc[i]['Close']

    def getDayPriceOpen(self, i):
        return self.testData.iloc[i]['Open']

    def trendDifferencing(self, timeseries, n):
        return timeseries-timeseries.shift(n)

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
        self.cat = 'Classification'
        
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
        
    def initMod(self, params, trainUpTo):
        self.params = params
        self.lag_n = params['lag']
        self.stock.resetData(self.lag_n)
        self.X = self.stock.differencedData[:trainUpTo]
        self.y = self.stock.data[self.cat][:trainUpTo]
        
    def validate(self, day, n_splits = 2, kfold = True):        
        kf = KFold(n_splits)
        dayBefore= day-datetime.timedelta(days=1)

        combinations = self.generateCombinations(self.param_ranges)
        bestParams = []
        bestScore = -100
        self.initMod(self.params, dayBefore)
        self.fit(self.X,self.y)
        if not kfold:
            return #If we aren't cross validating end here

        for combo in combinations:
            total = 0
            self.initMod(combo, dayBefore)
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                self.fit(X_train, y_train)
                total += self.score(X_test, y_test)

            if total/n_splits > bestScore:
                bestScore = total/n_splits
                bestParams = combo
        print("[info] model validated, chosing params: " + str(bestParams))
        self.initMod(bestParams, dayBefore)
        print('x: ' + str(self.X.head()))
        print('y: ' + str(self.y.head()))
        self.fit(self.X, self.y)

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

            predictY = self.mod.predict(self.stock.differencedData[day:day])
            actualY = self.stock.data[self.cat][day:day]
            print('predicty: ' + str(predictY))
            pYield = self.stock.declassify(predictY[0])

            pYields.append(pYield)

            predictedYs.append(predictY[0])
            actualYs.append(actualY)
        self.predictedYs = predictedYs
        self.actualYs = actualYs
        self.pYields = pYields

        return pYields
    
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

class MLPCLASS(Model):
    def __init__(self, stock, params = {'hidden layers': (100,), 'alpha':.0001, 'lag': 5}, param_ranges = {'alpha': np.logspace(-5,1, num=4), 'lag' : range(2,10,3), 'hidden layers': [(100,), (1000,), (500,)]}, debug=False):
        super(MLPCLASS, self).__init__(stock, params=params, param_ranges=param_ranges, debug=debug)
        self.classification = True
        self.a = params['alpha']
        self.hidden_layer = params['hidden layers']
        self.mod = MLPClassifier(hidden_layer_sizes = self.hidden_layer, alpha=self.a, max_iter=500)
        self.name = 'MLPCLASS'

    def __str__(self):
        return "Multi-Layer Perceptron Classifier"

    def fit(self, X, y):
        self.mod.fit(X, np.ravel(y))
