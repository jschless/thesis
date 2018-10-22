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

class Simulation:
    def __init__(self, stocks, models, timeFrame, principal, alphas=[1], validation_freq=0, train_length=-1, debug=False):
        self.timeFrame = timeFrame
        self.stocks = self.init_stocks(stocks, train_length)
        self.models = self.init_models(models, ver=debug)
        self.principal = principal
        self.accounts = self.init_accounts()
        self.validation_freq = validation_freq
        self.train_length = train_length
        self.alphas = alphas
        self.debug = debug

    def init_accounts(self):
        accts = {}
        for mod in self.models:
            accts[mod.name] = {}
        return accts
                
    def init_stocks(self, stocks, train_length):
        temp = []
        for stock in stocks:
            temp.append(Stock(stock, self.timeFrame, train_length = train_length))
        return temp
    
    def init_models(self, models, ver):
        temp = []
        for mod in models:
            if mod == 'LINREG':
                temp.append(Model(debug=ver))
            elif mod == 'DCA':
                temp.append(DCA(debug=ver))
            elif mod == 'LASSO':
                temp.append(LassoModel(debug=ver))
            elif mod == 'RIDGE':
                temp.append(RidgeModel(debug=ver))
            elif mod == 'MLP':
                temp.append(MLP(debug=ver))
            elif mod == 'ARIMA':
                temp.append(ARIMAModel(debug=ver))
        return temp

    def run(self):
        for mod in self.models:
            for stock in self.stocks:
                mod.addStock(stock)
                for a in self.alphas:
                    pYields = mod.getYields(self.validation_freq)
                    n_days = stock.n_days_test
                    dailyCap = self.principal/n_days
                    cash = 0
                    principal, acctStock = self.principal, 0
                    acctValue = principal
                    snapshots = []
                    investments = []
                    for i in range(n_days):
                        cash += dailyCap
                        principal -= dailyCap
                        py = pYields[i]
                        stockPrice = stock.getDayPriceOpen(i)
                        cash, acctStock, moneySpent = self.buyOrSell(py, cash, acctStock, stockPrice, alpha=a)
                        acctValue = cash+principal+acctStock*stockPrice
                        snapshots.append((stock.closeTestData.index[i], acctValue))
                        investments.append(moneySpent)
                    mod.addPerformance(stock, a, snapshots)
                    mod.addInvestments(stock, a, investments)
                    mod.addYield(stock, a, 100*(acctValue-self.principal)/self.principal)
                    print("Investing $" + str(self.principal) + " in " + stock.name + " using " + mod.name + ' with alpha=' +str(a)+  ' from ' + str(stock.startDate) + ' to ' + str(stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))
                self.accounts[mod.name][stock.name] = (snapshots, investments, a)
                
                
    def buyOrSell(self, py, cash, stock, price, alpha=1, beta=1):
        #print("percent yield: " + str(py))
        percent = py*alpha
        if py > 0:
            #print("buying $" + str(cash*py) + " at " + str(price))

            if percent > 1:
                if self.debug:
                    print("[warning] attempting to spend more cash than you have. Spending all")
                percent = 1
            return cash-percent*cash, stock+percent*cash/price, percent*cash
        else:
            #print("selling " + str(stock*py) + " shares at " + str(price))
            if percent < -1:
                if self.debug:
                    print("[warning] attempting to sell mroe stock than you have. Selling all")
                percent = -1
            return cash-py*alpha*stock*price, stock-py*alpha*stock, py*alpha*cash

    def performanceByStock(self):
        stockMap = {}
        for stock in self.stocks:
            stockMap[stock.name] = {}
            for model in self.models:
                stockMap[stock.name][model.name] = (model.performance[stock.name], model.investments[stock.name])
        return stockMap

    def alphaPlot(self):
        numModels = len(self.models)
        ax = plt.subplot(111)        
        beginXs = None
        beginInds = []
        offset = 0
        for model in self.models:
            for stock in model.stockList:
                yieldMap = model.yields[stock]
                xs = list(yieldMap.keys())
                inds = np.arange(len(yieldMap.keys()))
                if len(beginInds) == 0:
                    beginInds = inds
                    beginXs = xs
                ys = list(yieldMap.values())
                xs = [x + offset for x in xs]
                inds = [x + offset for x in inds]
                offset+=.1
                ax.bar(inds, ys, width = .1, align='center', label=model.name+"-"+stock)
        plt.title(r"Percent Yield as $\alpha$ Changes")
        plt.xlabel(r'$\alpha$')
        plt.ylabel("Percent Yield")
        plt.xticks(beginInds, beginXs)
        plt.legend()
        plt.show()

    def plotInvestmentAmounts(self):
        stockMap = self.performanceByStock()
        for stock, modelMap in stockMap.items():
            thisStock = next((x for x in self.stocks if x.name== stock), None)
            xs = thisStock.closeTestData.index.tolist()
            ys = thisStock.closeTestData['Close']
            fig, ax = plt.subplots(2, 1, sharex=True)
            for model, alphaMaps in modelMap.items():
                bestPerformance = (None, 0)
                bestInvestment = None
                for alpha, performance in alphaMaps[0].items():
                    investments = alphaMaps[1][alpha]
                    if performance[-1][1] > bestPerformance[1]:
                        bestPerformance = (performance, performance[-1][1])
                        bestInvestment = investments
                
                ax[0].plot(xs, bestInvestment, label=stock+"-"+model+"-alpha="+str(alpha))
                ax[1].plot(xs, ys)

                ax[0].set_title("Day-to-day investments")#invest $" + str(self.principal))# + " in " + stock + " using " + strat)
                ax[0].set(ylabel="Investment Amount")
                ax[1].set(xlabel= "Time", ylabel="Stock price")
            ax[0].legend()#loc='upper left')
            fig.autofmt_xdate()
        plt.show()
        
    def plotPerformanceByStock(self):
        stockMap = self.performanceByStock()
        for stock, modelMap in stockMap.items():
            thisStock = next((x for x in self.stocks if x.name== stock), None)
            xs = thisStock.closeTestData.index.tolist()
            ys = thisStock.closeTestData['Close']
            fig, ax = plt.subplots(2, 1, sharex=True)
            for model, alphaMaps in modelMap.items():
                bestPerformance = (None, 0)
                bestInvestment = None
                for alpha, performance in alphaMaps[0].items():
                    investments = alphaMaps[1][alpha]
                    if performance[-1][1] > bestPerformance[1]:
                        bestPerformance = (performance, performance[-1][1])
                        bestInvestment = investments
                
                ax[0].plot(*zip(*bestPerformance[0]), label=stock+"-"+model+"-alpha="+str(alpha))
                ax[1].plot(xs, ys)
                yieldColors = ['green' if i >= 0 else 'red' for i in bestInvestment]
                investments = [abs(x) for x in bestInvestment]
                if model == "DCA":
                    investments = [10 for x in investments]
                else:
                    investments = [2*100*x/max(investments) for x in investments] #scaling
                ax[0].scatter(*zip(*performance), s=investments, c=yieldColors)
                ax[0].set_title("Performance of investing $" + str(self.principal))# + " in " + stock + " using " + strat)
                ax[0].set(ylabel="Portfolio value")
                ax[1].set(xlabel= "Time", ylabel="Stock price")
            ax[0].legend()#loc='upper left')
            fig.autofmt_xdate()
        plt.show()

def tester():
    stocks = ['GOOG', 'AAPL']
    models = ['LASSO', 'LINREG', 'RIDGE', 'DCA']
    timeFrame = (datetime.date(2015,6,20), datetime.date(2015,8,20))
    principal = 35000 #amount of money starting the investment with
    zeroGainTimeFrame = (datetime.date(2013 ,12, 2), datetime.date(2014, 5, 12))
    #$531.48 -> $517.78
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = np.linspace(1,4,6)
    test = Simulation(stocks, models, zeroGainTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.plotPerformanceByStock()
    test.alphaPlot()

def marketCrash():
    stocks = ['GOOG']
    models = ['RIDGE', 'DCA']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2008,11,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = [1.0]#np.linspace(1,4,3)
    test = Simulation(stocks, models, negativeTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.plotInvestmentAmounts()
    test.plotPerformanceByStock()
    test.alphaPlot()
    
marketCrash()
