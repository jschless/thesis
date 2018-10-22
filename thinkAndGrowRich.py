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
        self.results = StratResults()
        self.results.stocks = self.stocks
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
                        #print('total acct value at day ' + str(i) + ' = ' + str(acctValue))
                        snapshots.append((stock.closeTestData.index[i], acctValue))
                        investments.append(moneySpent)
                    print("Investing $" + str(self.principal) + " in " + stock.name + " using " + mod.name + ' with alpha=' +str(a)+  ' from ' + str(stock.startDate) + ' to ' + str(stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))
                self.accounts[mod.name][stock.name] = (snapshots, investments, a)
                
                
    def buyOrSell(self, py, cash, stock, price, alpha=1, beta=1):
        #print("percent yield: " + str(py))
        percent = py*alpha
        if py > 0:
            #print("buying $" + str(cash*py) + " at " + str(price))

            if percent > 1:
                print("[warning] attempting to spend more cash than you have. Spending all")
                percent = 1
            return cash-percent*cash, stock+percent*cash/price, percent*cash
        else:
            #print("selling " + str(stock*py) + " shares at " + str(price))
            if percent < -1:
                print("[warning] attempting to sell mroe stock than you have. Selling all")
                percent = -1
            return cash-py*alpha*stock*price, stock-py*alpha*stock, py*alpha*cash

    def invertDict(self):
        inverted = {}
        for strat, s in self.accounts.items():
            for stock, data in s.items():
                strats = inverted.get(stock, {})
                strats[strat] = data
                inverted[stock] = strats
        return inverted

    def simplePlotInvestments(self, individual=True):
        stockToStrat = self.invertDict()
        for stock, s in stockToStrat.items():
            thisStock = None
            for stck in self.stocks:
                if stock == stck.name:
                    thisStock= stck
                    break
            xs = thisStock.closeTestData.index.tolist()
            ys = thisStock.closeTestData['Close']
            fig, ax = plt.subplots(2, 1, sharex=True)
            for strat, data in s.items():
                timeseries, investments, alpha = data
                ax[0].plot(*zip(*timeseries), label=stock+"-"+strat+"-alpha="+str(alpha))
                ax[1].plot(xs, ys)
                yieldColors = []
                investments
                for i in investments:
                    if i >= 0:
                        yieldColors.append('green')
                    else:
                        yieldColors.append('red')
                investments = [abs(x) for x in investments]
                investments = [2*100*x/max(investments) for x in investments]
                ax[0].scatter(*zip(*timeseries), s=investments, c=yieldColors)
                ax[0].set_title("Performance of investing $" + str(self.principal))# + " in " + stock + " using " + strat)
                ax[0].set(ylabel="Portfolio value")
                ax[1].set(xlabel= "Time", ylabel="Stock price")
            ax[0].legend()#loc='upper left')
            fig.autofmt_xdate()
        plt.show()
                    
    '''
    inputs:
    results: output from simulation function
    
    options:
    individualStocks: plot stock performance individually
    individualInvest: plot investment performance individually
    '''
    def visualize(self, individualStocks=False, individualInvest=False):
        self.simplePlotInvestments(individual=individualInvest)

class StratResults:
    def __init__(self):
        self.stocks = []
        self.performance = {} #stock-> ( alpha -> timeseries of investments)
        self.investments = {} #stock -> amount invested

    def addStock(self, stock):
        self.stocks.append(stock)

    def addPerformance(self, stock, performance):
        self.performance[stock] = performance

    def addInvestments(self, stock, investment):
        self.investments[stock] = investment

def tester():
    stocks = ['GOOG']
    models = ['LASSO', 'DCA']#, 'RIDGE', 'LINREG']#, 'LASSO', 'RIDGE']
    timeFrame = (datetime.date(2015,6,20), datetime.date(2015,8,20))
    principal = 35000
    zeroGainTimeFrame = (datetime.date(2013 ,12, 2), datetime.date(2014, 5, 12))
    #$531.48 -> $517.78
    validations = 10
    train_length = 100
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    test = Simulation(stocks, models, zeroGainTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.visualize(individualInvest=True)

tester()
