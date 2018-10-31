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
        self.name = self.generate_name()

    def generate_name(self):
        name = ""
        for s in self.stocks:
            name += s.name + ','
        name += '-'
        for m in self.models:
            name += m.name + ','
        name += '-' + str(self.timeFrame[0]) + '-' + str(self.timeFrame[1])
        name += '-alphas-' + str(self.alphas)
        return name

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
                    pYields = mod.getYields(self.validation_freq) #estimated percent yields
                    n_days = stock.n_days_test #number of days we're investing
                    dailyCap = self.principal/n_days 
                    principal, acctStock, cash = self.principal, 0, 0
                    acctValue = principal
                    snapshots, investments, cashStock = [], [], []
                    for i in range(n_days):
                        cash = dailyCap #get dailyCap in cash to spend for the day
                        principal -= dailyCap 
                        stockPrice = stock.getDayPriceOpen(i)
                        stockPriceClose = stock.getDayPriceClose(i)
                        cash, acctStock, moneySpent = self.buyOrSell(pYields[i], cash, acctStock, stockPrice, alpha=a)
                        principal += (dailyCap-moneySpent) #return money not spent to principal
                        acctValue = principal+acctStock*stockPriceClose #compute account value
                        snapshots.append(acctValue)
                        investments.append(moneySpent)
                        cashStock.append((cash, acctStock*stockPriceClose))
                        if self.debug:
                            print("[debug] percent yield: %f\n[debug] spent %f on %s at stock price %f\n[debug] account value is now %f" %(py, moneySpent, str(stock.closeTestData.index[i]), stockPriceClose, acctValue))
                            print("[debug] principal: %f     dailyCap:  %f    cash available:  %f   stock owned: %f" % (principal, dailyCap, cash, acctStock))                        
                    mod.addPerformance(stock, a, snapshots)
                    mod.addInvestments(stock, a, investments)
                    mod.addYield(stock, a, 100*(acctValue-self.principal)/self.principal)
                    mod.addCashStock(stock, a, cashStock)
                    print("[info] Investing $" + str(self.principal) + " in " + stock.name + " using " + mod.name + ' with alpha=' +str(a)+  ' from ' + str(stock.startDate) + ' to ' + str(stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))                
                
    def buyOrSell(self, py, cash, stock, price, alpha=1, beta=1):
        if py > 0:
            percent = py*alpha
            if debug:
                print("[info] buying $" + str(cash*py) + " at " + str(price))
            if percent > 1:
                if self.debug:
                    print("[warning] attempting to spend more cash than you have. Spending all")
                percent = 1
            return cash-percent*cash, stock+percent*cash/price, percent*cash
        else:
            percent = py*beta
            if debug:
                print("[info] selling " + str(stock*py) + " shares at " + str(price))
            if percent < -1:
                if self.debug:
                    print("[warning] attempting to sell more stock than you have. Selling all")
                percent = -1
            stockValue = percent*stock*price
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
        plt.set(title= r"Percent Yield as $\alpha$ Changes", xlabel = r'$\alpha$', ylabel = 'Percent Yield')
        plt.legend()
        plt.savefig('C:\\Users\\x92423\\Dropbox\\Research\\Math Thesis\\Results\\' + self.name + '-alphaplot.png')
        plt.show()
        
    def getDays(self, stock):
        for model in self.models:
            thisStock = next((x for x in self.stocks if x.name== stock), None)
            xs = thisStock.closeTestData.index.tolist()
            return xs
        
    def plotPredictedStockPerformance(self, stock, axis):
        for model in self.models:
            thisStock = next((x for x in self.stocks if x.name== stock), None)
            xs = thisStock.closeTestData.index.tolist()
            if not model.name == 'DCA':
                axis.plot(xs, model.predictedYs, label = 'Predicted Performance-' + model.name)
        axis.legend()

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
        axis.legend()
        
    def plotStockPerformance(self, stock, axis):
        stockMap = self.performanceByStock()
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        ys = thisStock.closeTestData['Close']
        axis.plot(xs, ys, label='Actual Performance')
        axis.legend()
        
    def plotInvestmentAmount(self, stock, axis): 
        stockMap = self.performanceByStock()       
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        modelMap = stockMap[stock]
        dca = False
        for model, alphaMaps in modelMap.items():
            for alpha, performance in alphaMaps[0].items():
                if model == 'DCA':
                    if not dca:
                        investments = alphaMaps[1][alpha]
                        axis.scatter(xs, investments, s=10, label=model)
                        dca = True
                else:
                    investments = alphaMaps[1][alpha]
                    zeroLine = [0 for x in xs]
                    axis.plot(xs, zeroLine, '--')
                    axis.scatter(xs, investments, s=10, label=model+"-alpha-"+str(alpha))
        axis.legend()

    def plotPortfolioAmount(self, stock, axis): 
        stockMap = self.performanceByStock()       
        thisStock = next((x for x in self.stocks if x.name== stock), None)
        xs = thisStock.closeTestData.index.tolist()
        modelMap = stockMap[stock]
        dca = False
        for model, alphaMaps in modelMap.items():
            for alpha, performance in alphaMaps[0].items():
                if model=='DCA':
                    if not dca: 
                        axis.plot(xs, performance, label=model)
                        dca = True
                else:
                    axis.plot(xs, performance, label=model+"-alpha-"+str(alpha))
        axis.legend()


    def plotStuff(self):
        stockMap = self.performanceByStock()
        plt.figure(figsize=(18,10))
        for i in range(len(self.stocks)):
            stock = self.stocks[i]
            fig, ax = plt.subplots(3,1,figsize=(16,10), sharex=True)
            self.plotPredictedStockPerformance(stock.name,ax[2])
            self.plotStockPerformance(stock.name, ax[2])
            self.plotPortfolioAmount(stock.name, ax[0])
            self.plotInvestmentAmount(stock.name, ax[1])
            ax[2].set(xlabel= "Time", ylabel="Stock Price ($)", title='Predicted v. Actual Stock Performance')
            ax[1].set(ylabel= "Value ($)", title='Investment and Liquidation Amounts')
            ax[0].set(ylabel= "Value ($)", title='Portfolio Performance')
            fig.autofmt_xdate()
            fig.tight_layout()
            for j in range(3):
                ax[j].set_xticks(self.getDays(stock.name), minor=True)
                ax[j].grid(which='minor', alpha=.2)
                ax[j].grid(which='major', alpha=.5)

        plt.savefig('C:\\Users\\x92423\\Dropbox\\Research\\Math Thesis\\Results\\' + self.name + '-performanceplot.png')
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
    stocks = ['GOOG']
    models = ['RIDGE', 'DCA']
    negativeTimeFrame = (datetime.date(2007,11,1), datetime.date(2008,11,1))
    #GOOG: $344.26 -> $145.53
    principal = 35000 #amount of money starting the investment with
    validations = 10 #validate hyperparameters every n_days
    train_length = 100
    alphas = [10000000]
    zeroGainTimeFrame = (datetime.date(2013 ,12, 2), datetime.date(2014, 5, 12))
    test = Simulation(stocks, models, zeroGainTimeFrame, principal, validation_freq=validations, train_length = train_length, debug=False, alphas=alphas)
    test.run()
    test.plotStuff()
simpleTest()

    
#alphaTester()
