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
    def __init__(self, stocks, models, timeFrame, principal, validation_freq=0, train_length=-1):
        self.timeFrame = timeFrame
        self.stocks = self.init_stocks(stocks, train_length)
        self.models = self.init_models(models)
        self.principal = principal
        self.accounts = self.init_accounts()
        self.validation_freq = validation_freq
        self.train_length = train_length
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
    
    def init_models(self, models):
        temp = []
        for mod in models:
            if mod == 'LINREG':
                temp.append(Model())
            elif mod == 'DCA':
                temp.append(DCA())
            elif mod == 'LASSO':
                temp.append(LassoModel())
            elif mod == 'RIDGE':
                temp.append(RidgeModel())
            elif mod == 'MLP':
                temp.append(MLP())
            elif mod == 'ARIMA':
                temp.append(ARIMAModel())
        return temp

    def run(self):
        for mod in self.models:
            for stock in self.stocks:
                mod.addStock(stock)
                pYields = mod.getYields(self.validation_freq)
                n_days = stock.n_days_test
                dailyCap = self.principal/n_days
                cash = 0
                principal, acctStock = self.principal, 0
                acctValue = principal
                snapshots = []
                for i in range(n_days):
                    cash += dailyCap
                    #print("daily cap is " + str(cash))
                    principal -= dailyCap
                    py = pYields[i]
                    stockPrice = stock.getDayPriceClose(i)
                    #print('day ' + str(i))
                    cash, acctStock = self.buyOrSell(py, cash, acctStock, stockPrice)
                    acctValue = cash+principal+acctStock*stockPrice
                    #print('total acct value at day ' + str(i) + ' = ' + str(acctValue))
                    snapshots.append((stock.closeTestData.index[i], acctValue))
                print("Investing $" + str(self.principal) + " in " + stock.name + " using " + mod.name + ' from ' + str(stock.startDate) + ' to ' + str(stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))
                self.accounts[mod.name][stock.name] = snapshots
                
                
    def buyOrSell(self, py, cash, stock, price):
        #print("percent yield: " + str(py))
        if py > 0:
            #print("buying $" + str(cash*py) + " at " + str(price))
            return cash-py*cash, stock+py*cash/price
        else:
            #print("selling " + str(stock*py) + " shares at " + str(price))
            return cash-py*stock*price, stock-py*stock

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
            for strat, data in s.items():
                timeseries = data
                #plt.scatter(*zip(*timeseries))#, label=stock+"-"+strat)
                plt.plot(*zip(*timeseries), label=stock+"-"+strat)
                plt.title("Performance of investing $" + str(self.principal))# + " in " + stock + " using " + strat)
                plt.ylabel("Portfolio value")
                plt.xlabel("Time")
                plt.legend()#loc='upper left')
                if individual:
                    plt.figure()                 
     
    def simplePlotStocks(self, individual=True):
        for stock in self.stocks:
            xs = []
            ys = []
            for i in range(stock.n_days_test):
                xs.append(stock.startDate+datetime.timedelta(days=i))
                ys.append(stock.getDayPriceClose(i))
                #plt.scatter(xs, ys)
                plt.plot(xs, ys, label=stock.name)
                plt.title("Time Series of " + stock.name + " from " + str(stock.startDate) + " to " + str(stock.endDate))
                plt.xlabel("Date")
                plt.ylabel("Closing price")
                plt.show()
            if individual:
                plt.figure()
            else:
                plt.legend()#loc='upper left')
        
    '''
    inputs:
    results: output from simulation function
    
    options:
    individualStocks: plot stock performance individually
    individualInvest: plot investment performance individually
    '''
    def visualize(self, individualStocks=False, individualInvest=False):
        self.simplePlotInvestments(individual=individualInvest)
        plt.show()
       # plt.figure()
       # self.simplePlotStocks()
        
def tester():
    stocks = ['AAPL']
    models = ['LASSO', 'DCA']#, 'LASSO', 'RIDGE']
    timeFrame = (datetime.date(2009,6,20), datetime.date(2009,8,20))
    principal = 35000
    validations = 10
    train_length = 100
    test = Simulation(stocks, models, timeFrame, principal, validation_freq=validations, train_length = train_length)
    test.run()
    test.visualize()

tester()
