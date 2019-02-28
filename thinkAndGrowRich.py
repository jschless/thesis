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
from matplotlib import collections as mc

class Simulation:
    def __init__(self, stock, models, timeFrame, principal, alphas=[1],
                 validation_freq=0, train_length=-1, difference=1, debug=False, otherStocks=None):
        """Creates a simulation object.

        Keyword arguments:
        stock - symbol of S&P500 stock being explored 
        models - list of models being used (string codes)
        timeFrame - tuple of datetimes (start, end)
        principal - amount of money to begin investing with
        alphas = list of alphas, determines how bold to be with investing
        validation_freq - how often to validate hyperparameters
        train_length - if you want to limit the amount of data you train on.
        """
        self.debug = debug
        self.timeFrame = timeFrame
        self.difference = difference
        self.stock = Stock(stock, self.timeFrame, train_length = train_length, difference=difference, otherStocks=otherStocks)
        self.models = self.init_models(models)
        self.principal = principal
        self.validation_freq = validation_freq
        self.train_length = train_length
        self.alphas = alphas
        self.stockYield = self.stock.getYield()
        self.name = self.generate_name()
        self.markers = itertools.cycle(('.', 'v', '1', '3', '<', 's', '*', '^',  '3', '4', 'p', '+', 'd'))
        self.colorMap = {'DCA': 'b', 'MLP': 'g', 'LINREG': 'r', 'RIDGE': 'c', 'LASSO':'m', 'MLPCLASS': 'r'}

    def generate_name(self):
        """Creates a unique name for the simulation ran"""
        name = self.stock.name + '-'
        for mod in self.models:
            name += mod.name + '-'
        name += str(self.timeFrame[0]) + '--' + str(self.timeFrame[1]) + '-'
#        name += 'alphas--' + str(self.alphas)
#        name += '-train_length-' + str(self.train_length)
        return name
                    
    def init_models(self, models):
        """Initializes models"""
        temp = []
        for mod in models:
            if mod == 'LINREG':
                temp.append(Model(self.stock,debug=self.debug))
            elif mod == 'DCA':
                temp.append(DCA(self.stock, debug=self.debug))
            elif mod == 'LASSO':
                temp.append(LassoModel(self.stock,debug=self.debug))
            elif mod == 'RIDGE':
                temp.append(RidgeModel(self.stock,debug=self.debug))
            elif mod == 'MLP':
                temp.append(MLP(self.stock,debug=self.debug))
            elif mod == 'MLPCLASS':
                temp.append(MLPCLASS(self.stock, debug=self.debug))
        return temp

    def run(self):
        """Runs simulation on stock over time period"""
        for mod in self.models:
            pYields = mod.getYields(self.validation_freq) #estimated percent yields
            for a in self.alphas:
                n_days = self.stock.n_days_test #number of days we're investing
                daysRemaining = n_days
                principal, acctStock, cash = self.principal, 0, 0
                acctValue = principal
                snapshots, investments, cashStock = [], [], []
                dailyCap = self.principal/(daysRemaining)
                for i in range(n_days):

                    if mod.name == 'DCA':
                        cash = dailyCap #get dailyCap in cash to spend for the day
                    else:
                        cash = dailyCap*5
                    #principal -= dailyCap 
                    stockPrice = self.stock.getDayPriceOpen(i)
                    stockPriceClose = self.stock.getDayPriceClose(i)
                    stockCap = acctStock/(daysRemaining+10 )
                    #acctStock -= stockCap
                    #cash, acctStock, moneySpent, stockSold = self.buyOrSell(pYields[i], cash, stockCap, stockPrice, alpha=a, beta=a)
                    stockChange, moneyChange,  = self.buyOrSell(pYields[i], cash, stockCap, stockPrice, alpha=a, beta=a)
                    principal += moneyChange
                    acctStock += stockChange

                    acctValue = principal+acctStock*stockPriceClose #compute account value

                    snapshots.append(acctValue)
                    investments.append(moneyChange)
                    cashStock.append((cash, acctStock*stockPriceClose))
                    daysRemaining -= 1
                mod.addPerformance(a, snapshots)
                mod.addInvestments(a, investments)
                mod.addYield(a, 100*(acctValue-self.principal)/self.principal)
                mod.addCashStock(a, cashStock)
                print("[info] Investing $" + str(self.principal) + " in " + self.stock.name + " using " + mod.name + ' with alpha=' +str(a)+  ' from ' + str(self.stock.startDate) + ' to ' + str(self.stock.endDate) + ' yielded %' + str(100*(acctValue-self.principal)/self.principal))

                print("[info] Stock yield over timeframe: " + str(self.stockYield))
                
    def buyOrSell(self, py, cash, stock, price, alpha=1, beta=1):
        """buys or sells stocks, depending on estimated percent yield
        
        Keyword arguments:
        py - proportion of money to invest
        cash - amount available to spend
        stock - amount of stock available (to sell)
        price - price of stock at the moment
        alpha - parameter to scale amount of money to spend purchasing
        beta - parameter to scale amount of stock to sell
        """
        if py > 0:
            percent = py*alpha
            if percent > 1:
                print("[warning] attempting to spend more cash than you have. Spending all")
                percent = 1
            stockChange = percent*cash/price
            cashChange = -percent*cash
            return stockChange, cashChange
        else:
            percent = py*beta
            if percent < -1:
                print("[warning] attempting to sell more stock than you have. Selling all")
                percent = -1
            stockChange = percent*stock
            cashChange = -percent*stock*price
            return stockChange, cashChange



# --------------------------------------------------------------------------------------- #
# ------------------------------------- PLOTTING SECTION -------------------------------- #
# --------------------------------------------------------------------------------------- #


    def alphaPlot(self):
        numModels = len(self.models)
        ax = plt.subplot(111)        
        beginXs = None
        beginInds = []
        offset = 0
        for model in self.models:
            yieldMap = model.yields[self.stock.name]
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
        
    def getDays(self):
        return self.stock.testData.index.tolist()
        
    def plotPredictedStockPerformance(self, axis):
        for model in self.models:
            if not model.name == 'DCA' and not model.classification:
                axis.plot(self.getDays(), model.predictedYs, label = 'Predicted Performance-' + model.name, marker = next(self.markers), color=self.colorMap[model.name])
        axis.legend()

    def plotCashStock(self, axis):
        for model in self.models:
            for alpha, cashStock in model.cashStock.items():
                cash = [x[0] for x in cashStock]
                stocks = [x[1] for x in cashStock]
                axis.plot(xs, cash, label="cash-"+model.name+"alpha="+str(alpha), color=self.colorMap[model.name])
                axis.plot(xs, stocks, label="stock-"+model.name+"alpha="+str(alpha), color=self.colorMap[model.name])
        axis.legend()
        
    def plotStockPerformance(self,  axis, cat='Close'):
        axis.plot(self.getDays(), self.stock.testData['Close'], label='Actual Performance', marker = next(self.markers))
        axis.legend()

    def plotActualToPredicted(self, axis, cat='Close'):
        days = self.getDays()
        for model in self.models:
            if not model.name == 'DCA' and not model.classification:
                for i in range(len(days)-1):
                    axis.plot([days[i], days[i+1]], [self.stock.testData.iloc[i]['Close'], model.predictedYs[i+1]], 'r--')

    def plotInvestmentAmount(self, axis): 
        dca = False
        for model in self.models:
            for alpha, investments in model.investments.items():
                if model.name == 'DCA':
                    if not dca:
                        axis.scatter(self.getDays(), investments, s=10, label=model.name, marker = next(self.markers), color=self.colorMap[model.name])
                        dca = True
                else:
                    zeroLine = [0 for x in self.getDays()]
                    axis.plot(self.getDays(), zeroLine, '--')
                    axis.scatter(self.getDays(), investments, s=10, label=model.name+"-alpha-"+str(alpha), marker = next(self.markers), color=self.colorMap[model.name])
        axis.legend()

    def plotPortfolioAmount(self, axis): 
        dca = False
        for model in self.models:
            for alpha, performance in model.performance.items():
                if model.name=='DCA':
                    if not dca: 
                        axis.plot(self.getDays(), performance, label=model.name, marker = next(self.markers), color=self.colorMap[model.name])
                        dca = True
                else:
                    axis.plot(self.getDays(), performance, label=model.name+"-alpha-"+str(alpha), marker = next(self.markers), color=self.colorMap[model.name])
        axis.legend()

    def plotStuff(self):
        stock = self.stock
        fig, ax = plt.subplots(3,1,figsize=(16,10), sharex=True)
        self.plotPredictedStockPerformance(ax[2])
        self.plotStockPerformance(ax[2], cat='Open')
        #self.plotActualToPredicted(ax[2], cat='Open')
        self.plotPortfolioAmount(ax[0])
        self.plotInvestmentAmount(ax[1])
        ax[2].set(xlabel= "Time", ylabel="Stock Price ($)", title='Predicted v. Actual Stock Performance')
        ax[1].set(ylabel= "Value ($)", title='Investment and Liquidation Amounts')
        ax[0].set(ylabel= "Value ($)", title='Portfolio Performance')
        fig.autofmt_xdate()
        fig.tight_layout()
        for j in range(3):
            ax[j].set_xticks(self.getDays(), minor=True)
            ax[j].grid(which='minor', alpha=.2)
            ax[j].grid(which='major', alpha=.5)

        plt.savefig('C:\\Users\\x92423\\Dropbox\\Research\\Math Thesis\\Results\\' + self.name + '-performanceplot.png')
        plt.show()
