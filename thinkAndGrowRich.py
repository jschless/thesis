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
                    cashStock = []
                    for i in range(n_days):
                        cash += dailyCap
                        principal -= dailyCap
                        py = pYields[i]
                        stockPrice = stock.getDayPriceOpen(i)
                        stockPriceClose = stock.getDayPriceClose(i)
                        cash, acctStock, moneySpent = self.buyOrSell(py, cash, acctStock, stockPrice, alpha=a)                        
                        acctValue = cash+principal+acctStock*stockPriceClose
                        snapshots.append((stock.closeTestData.index[i], acctValue))
                        cashStock.append((cash, acctStock*stockPriceClose))
                        if self.debug:
                            print("[debug] percent yield: %f\n[debug] spent %f on %s at stock price %f\n[debug] account value is now %f" %(py, moneySpent, str(stock.closeTestData.index[i]), stockPriceClose, acctValue))
                            print("[debug] principal: %f     dailyCap:  %f    cash available:  %f   stock owned: %f" % (principal, dailyCap, cash, acctStock))
                        
                        investments.append(moneySpent)
                    mod.addPerformance(stock, a, snapshots)
                    mod.addInvestments(stock, a, investments)
                    mod.addYield(stock, a, 100*(acctValue-self.principal)/self.principal)
                    mod.addCashStock(stock, a, cashStock)
                    print("[info] Investing $" + str(self.principal) + " in " + stock.name + " using " + mod.name + ' with alpha=' +str(a)+  ' from ' + str(stock.startDate) + ' to ' + str(stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))                
                
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
            stockValue = py*alpha*stock*price
            numShares = stockValue/price
            return cash-stockValue, stock+numShares, stockValue

    def performanceByStock(self):
        stockMap = {}
        for stock in self.stocks:
            stockMap[stock.name] = {}
            for model in self.models:
                stockMap[stock.name][model.name] = (model.performance[stock.name], model.investments[stock.name], model.cashStock[stock.name])
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

    def plotPredictedStockPerformance(self, stock, axis):
        for model in self.models:
            thisStock = next((x for x in self.stocks if x.name== stock), None)
            xs = thisStock.closeTestData.index.tolist()
            axis.plot(xs, model.predictedYs, label = model.name)
        axis.set(title = "Predicted Stock Performance", xlabel = 'Day', ylabel='Price')

    def plotCashStock(self, stock, axis):
        stockMap = self.performanceByStock()       
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        modelMap = stockMap[stock]
        for model, alphaMaps in modelMap.items():
            for alpha, performance in alphaMaps[0].items():
                cashStock = alphaMaps[2][alpha]
                cash = [x[0] for x in cashStock]
                stocks = [x[1] for x in cashStock]
                axis.plot(xs, cash, label="cash-"+model+"alpha="+str(alpha))
                axis.plot(xs, stocks, label=stock+"-"+model+"alpha="+str(alpha))
        axis.set(xlabel= "Time", ylabel="Dollar Amount", title = "Cash v. Stock")
        
    def plotStockPerformance(self, stock, axis):
        stockMap = self.performanceByStock()
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        ys = thisStock.closeTestData['Close']
        axis.plot(xs, ys, label=stock)
        axis.set(xlabel= "Time", ylabel="Stock price", title = stock + " Performance")
        
    def plotInvestmentAmount(self, stock, axis): 
        stockMap = self.performanceByStock()       
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        modelMap = stockMap[stock]
        for model, alphaMaps in modelMap.items():
            for alpha, performance in alphaMaps[0].items():
                investments = alphaMaps[1][alpha]
                axis.plot(xs, investments, label=stock+"-"+model+"alpha="+str(alpha))
        axis.set(xlabel= "Time", ylabel="Dollar Investment", title = "Amount Invested")

    def plotPortfolioAmount(self, stock, axis): 
        stockMap = self.performanceByStock()       
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        modelMap = stockMap[stock]
        for model, alphaMaps in modelMap.items():
            for alpha, performance in alphaMaps[0].items():
                axis.plot(*zip(*performance), label=stock+"-"+model+"alpha="+str(alpha))
        axis.set(xlabel = "Time", ylabel = "Portfolio Value", title="Portfolio Value Over Time")

    def plotGenerator(self):
        stockMap = self.performanceByStock()
        for i in range(len(self.stocks)):
            fig, ax = plt.subplots(4,1, sharex=True)
            #plt.figure(i+1)
            stock = self.stocks[i]
            self.plotPortfolioAmount(stock.name, ax[0])
            self.plotInvestmentAmount(stock.name, ax[1])
            self.plotStockPerformance(stock.name, ax[2])
            #self.plotPredictedStockPerformance(stock.name, ax[3])
            self.plotCashStock(stock.name, ax[3])
            fig.autofmt_xdate()
            for j in range(4):
                ax[j].legend()

        plt.show()
            
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


marketCrash()
