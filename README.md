# Machine-Learning-Stock-Predictions
Machine Learning Stock Predictions is designed to be an intuitive and extensible project that harnesses machine learning to make stock predictions. 

Initally, I clean and prepare a dataset of historical stock prices (from 31/12/2017 to 31/12/2018) and fundamental infomration using pandas, after which I apply a scikit-learn classifier to uncover the relationship between stock fundamentals (e.g. PE ratio, debt/equity, float, etc.) and the subsequent annual price change compared with the an index. I then conduct a backtest, before generating predictions on current data.

Note: Machine Learning Stock Predictions predicts which stocks will outperform the S&P500; however, it does not suggest how best to combine stocks into a portfolio. 

The overall workflow to use machine learning to make stocks prediction is as follows:

1. Acquire historical fundamental data – these are the features or predictors
2. Acquire historical stock price data – this is will make up the dependent variable, or label (what we are trying to predict).
3. Preprocess data
4. Use a machine learning model to learn from the data
5. Backtest the performance of the machine learning model
6. Acquire current fundamental data
7. Generate predictions from current fundamental data

Preliminaries 

This project uses python 3.6, and the common data science libraries pandas and scikit-learn. 

Historical Data

Data acquisition and preprocessing is probably the hardest part of most machine learning projects. But it is a necessary evil, so it's best to not fret and just carry on.

For this project, we need three datasets:

Historical stock fundamentals
Historical stock prices
Historical S&P500 prices
We need the S&P500 index prices as a benchmark: a 5% stock growth does not mean much if the S&P500 grew 10% in that time period, so all stock returns must be compared to those of the index.

Historical stock fundamentals
Historical fundamental data is actually very difficult to find (for free, at least). Although sites like Quandl do have datasets available, you often have to pay a pretty steep fee.

It turns out that there is a way to parse this data, for free, from Yahoo Finance. I will not go into details, because Sentdex has done it for us. On his page you will be able to find a file called intraQuarter.zip, which you should download, unzip, and place in your working directory. Relevant to this project is the subfolder called _KeyStats, which contains html files that hold stock fundamentals for all stocks in the S&P500 between 2003 and 2013, sorted by stock. However, at this stage, the data is unusable – we will have to parse it into a nice csv file before we can do any ML.

Historical price data
In the first iteration of the project, I used pandas-datareader, an extremely convenient library which can load stock data straight into pandas. However, after Yahoo Finance changed their UI, datareader no longer worked, so I switched to Quandl, which has free stock price data for a few tickers, and a python API. However, as pandas-datareader has been fixed, we will use that instead.

Likewise, we can easily use pandas-datareader to access data for the SPY ticker. Failing that, one could manually download it from yahoo finance, place it into the project directory and rename it sp500_index.csv.

The code for downloading historical price data can be run by entering the following into terminal:

python download_historical_prices.py

