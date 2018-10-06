import csv
import sys
import time
symbols = []
industries = []
dates = []
with open("s&p500Cos.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    #next(data) #skip header row
    for row in data:
        date = time.strptime(row[6], "%m/%d/%Y")
        dates.append(date)
        industries.append(row[3])
        symbols.append(row[0])

from subprocess import call
flag = False
for i in range(len(symbols)):
	if flag:
		call(["python", "apiYahoo3.py", "--symbol=" + symbols[i], "--from=1900-1-13", "--to=2018-08-23", "-o", symbols[i] + ".csv"])
	if symbols[i] == "MYL":
		flag = True