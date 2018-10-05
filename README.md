# Math Thesis
This is a git repository for my thesis on "Statistical Learning in Financial Markets"

# Files By Type

## Stock Simulation Code
- thinkAndGrowRich.py    This is the main file for the code base. It contains the code to run a simulation
- models.py    This creates the different statistical models and contains the stock object, which holds the data for the simulation

## Products for Turn-in
- litReview* Files pertaining to the literature review for the project
- Schlessinger_ResearchProposal.docx    The research proposal for the project

## Data Acquisition
- dataMunch.py Runs and pulls data for all stocks in the S&P 500 by accessing the Yahoo api. This pulls all available data up to the present.
- apiYahoo3.py Script that pulls requested stock data from Yahoo stock database
- s&p500Cos.csv List of all S&P 500 companies. This helps dataMunch.py pull in the data
