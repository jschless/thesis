import random
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import datetime
from  thinkAndGrowRich import *

def massiveTester():
    companies = pd.read_csv('C:\\Users\\x92423\\Documents\\Thesis Data Grab\\s&p500Cos.csv', encoding='windows-1252')
    #coList = [x for company in companies.iloc[:, [0]]]
    coList = ['GOOG']
    timeFrames = itertools.cycle((60, 365, 730, 1460, 2920))
    models = ['LINREG', 'DCA']#, 'LASSO', 'RIDGE']#, 'MLP']
    principal = 35000
    validations = 0
    alphas = [1]
    sims = []
    numTrials = 5
    
    toPlot = []
    modelPlot = {}
    stockYields = []
    for m in models:
        modelPlot[m] = []
    for i in range(5):
        for stock in coList:
            start = getRandomDate()
            end = start + datetime.timedelta(days=next(timeFrames))
            testTimeFrame = (start, end)
            print('test time frame: ' + str(testTimeFrame))
            simulation = Simulation(stock, models, testTimeFrame, principal, validation_freq=validations, debug=False, alphas=alphas)
            simulation.run()
            sims.append(simulation)
        modelMap = {}
        for sim in sims:
            for mod in sim.models:
                modelMap[mod.name] = None
                for alpha in sim.alphas:
                    if modelMap[mod.name] == None:
                        modelMap[mod.name] = mod.yields[alpha]
                    elif modelMap[mod.name] < mod.yields[alpha]:
                        modelMap[mod.name] = mod.yields[alpha]
            stockStart = sim.stock.testData['Open'].iloc[0]
            stockEnd = sim.stock.testData['Close'].iloc[-1]
            stockYield = (stockEnd-stockStart)/stockStart
            toPlot.append((stockYield, modelMap))
            stockYields.append(stockYield)
    for pyield, modelmap in toPlot:
        labels = list(modelmap.keys())
        yields = list(modelmap.values())
        print(labels)
        print(yields)
        for i in range(len(yields)):
            modelPlot[labels[i]].append(yields[i])
    for model, yields in modelPlot.items():
        plt.scatter(stockYields, yields, label = model)

    plt.xlabel('Stock Yield over Timeframe')
    plt.ylabel('Investment Strategy Yield')
    plt.title('Investment Performance v. Stock Performance')
    plt.legend()
    plt.show()
        
def getRandomDate():
    start = datetime.date(2005,1,1)
    delta = random.randint(0, 6000)
    return start + datetime.timedelta(days=delta)

massiveTester()
