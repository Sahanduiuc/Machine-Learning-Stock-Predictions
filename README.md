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

# Historical Data
For this project, I use three datasets:

  1. Historical stock fundamentals
  2. Historical stock prices
  3. Historical S&P500 prices

# Historical Stock Fundamentals
Historical fundamental data is actually very difficult to find (for free, at least). Although sites like Quandl do have datasets available, you often have to pay a pretty steep fee.

It turns out that there is a way to parse this data, for free, from Yahoo Finance. I will not go into details, because Sentdex has done it for us. On his page you will be able to find a file called intraQuarter.zip, which you should download, unzip, and place in your working directory. Relevant to this project is the subfolder called _KeyStats, which contains html files that hold stock fundamentals for all stocks in the S&P500 between 2003 and 2013, sorted by stock. However, at this stage, the data is unusable – we will have to parse it into a nice csv file before we can do any ML.

# Historical Price Data
In the first iteration of the project, I used pandas-datareader, an extremely convenient library which can load stock data straight into pandas. However, after Yahoo Finance changed their UI, datareader no longer worked, so I switched to Quandl, which has free stock price data for a few tickers, and a python API. However, as pandas-datareader has been fixed, we will use that instead.

Likewise, we can easily use pandas-datareader to access data for the SPY ticker. Failing that, one could manually download it from yahoo finance, place it into the project directory and rename it sp500_index.csv.

The code for downloading historical price data can be run by entering the following into terminal:

python download_historical_prices.py

# Creating the Training Dataset
Our ultimate goal for the training data is to have a 'snapshot' of a particular stock's fundamentals at a particular time, and the subsequent annual performance of the stock.

For example, if our 'snapshot' consists of all of the fundamental data for AAPL on the date 28/1/2017, then we also need to know the percentage price change of AAPL between 28/1/17 and 28/1/18. Thus, our algorithm can learn how the fundamentals impact the annual change in the stock price.

In fact, this is a slight oversimplification. In fact, what the algorithm will eventually learn is how fundamentals impact the outperformance of a stock relative to the S&P500 index. This is why we also need index data.

# Preprocessing Historical Price Data
When pandas-datareader downloads stock price data, it does not include rows for weekends and public holidays (when the market is closed).

However, referring to the example of AAPL above, if our snapshot includes fundamental data for 28/1/05 and we want to see the change in price a year later, we will get the nasty surprise that 28/1/2006 is a Saturday. Does this mean that we have to discard this snapshot?

By no means – data is too valuable to callously toss away. As a workaround, I instead decided to 'fill forward' the missing data, i.e we will assume that the stock price on Saturday 28/1/2006 is equal to the stock price on Friday 27/1/2006.

# Features
Below is a list of some of the interesting variables that are available on Yahoo Finance.

# Valuation Measures
<ul>
<li>'Market Cap'</li>
<li>Enterprise Value</li>
<li>Trailing P/E</li>
<li>Forward P/E</li>
<li>PEG Ratio</li>
<li>Price/Sales</li>
<li>Price/Book</li>
<li>Enterprise Value/Revenue</li>
<li>Enterprise Value/EBITDA</li>
</ul>
# Financials
<ul>
<li>Profit Margin</li>
<li>Operating Margin</li>
<li>Return on Assets</li>
<li>Return on Equity</li>
<li>Revenue</li>
<li>Revenue Per Share</li>
<li>Quarterly Revenue Growth</li>
<li>Gross Profit</li>
<li>EBITDA</li>
<li>Net Income Avi to Common</li>
<li>Diluted EPS</li>
<li>Quarterly Earnings Growth</li>
<li>Total Cash</li>
<li>Total Cash Per Share</li>
<li>Total Debt</li>
<li>Total Debt/Equity</li>
<li>Current Ratio</li>
<li>Book Value Per Share</li>
<li>Operating Cash Flow</li>
<li>Levered Free Cash Flow</li>
</ul>
# Trading Information
<ul>
<li>Beta</li>
<li>50-Day Moving Average</li>
<li>200-Day Moving Average</li>
<li>Avg Vol (3 month)</li>
<li>Shares Outstanding</li>
<li>Float</li>
<li>% Held by Insiders</li>
<li>% Held by Institutions</li>
<li>Shares Short</li>
<li>Short Ratio</li>
<li>Short % of Float</li>
<li>Shares Short (prior month)</li>
</ul>
# Parsing
However, all of this data is locked up in HTML files. Thus, we need to build a parser. In this project, I did the parsing with regex, but please note that generally it is really not recommended to use regex to parse HTML. However, I think regex probably wins out for ease of understanding (this project being educational in nature), and from experience regex works fine in this case.

This is the exact regex used:

<code>r'>' + re.escape(variable) + r'.*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0|NaN)%?(</td>|</span>)'</code>

While it looks pretty arcane, all it is doing is searching for the first occurence of the feature (e.g "Market Cap"), then it looks forward until it finds a number immediately followed by a </td> or </span> (signifying the end of a table entry). The complexity of the expression above accounts for some subtleties in the parsing:

the numbers could be preceeded by a minus sign
Yahoo Finance sometimes uses K, M, and B as abbreviations for thousand, million and billion respectively.
some data are given as percentages
some datapoints are missing, so instead of a number we have to look for "N/A" or "NaN.
Both the preprocessing of price data and the parsing of keystats are included in parsing_keystats.py. Run the following in your terminal:

python parsing_keystats.py
You should see the file keystats.csv appear in your working directory. Now that we have the training data ready, we are ready to actually do some machine learning.

Backtesting
Backtesting is arguably the most important part of any quantitative strategy: you must have some way of testing the performance of your algorithm before you live trade it.

Despite its importance, I originally did not want to include backtesting code in this repository. The reasons were as follows:

Backtesting is messy and empirical. The code is not very pleasant to use, and in practice requires a lot of manual interaction.
Backtesting is very difficult to get right, and if you do it wrong, you will be deceiving yourself with high returns.
Developing and working with your backtest is probably the best way to learn about machine learning and stocks – you'll see what works, what doesn't, and what you don't understand.
Nevertheless, because of the importance of backtesting, I decided that I can't really call this a 'template machine learning stocks project' without backtesting. Thus, I have included a simplistic backtesting script. Please note that there is a fatal flaw with this backtesting implementation that will result in much higher backtesting returns. It is quite a subtle point, but I will let you figure that out :)

Run the following in terminal:

python backtesting.py
You should get something like this:

Classifier performance
======================
Accuracy score:  0.81
Precision score:  0.75

Stock prediction performance report
===================================
Total Trades: 177
Average return for stock predictions:  37.8 %
Average market return in the same period:  9.2%
Compared to the index, our strategy earns  28.6 percentage points more
Again, the performance looks too good to be true and almost certainly is.

Current fundamental data
Now that we have trained and backtested a model on our data, we would like to generate actual predictions on current data.

As always, we can scrape the data from good old Yahoo Finance. My method is to literally just download the statistics page for each stock (here is the page for Apple), then to parse it using regex as before.

In fact, the regex should be almost identical, but because Yahoo has changed their UI a couple of times, there are some minor differences. This part of the projet has to be fixed whenever yahoo finance changes their UI, so if you can't get the project to work, the problem is most likely here.

Run the following in terminal:

python current_data.py
The script will then begin downloading the HTML into the forward/ folder within your working directory, before parsing this data and outputting the file forward_sample.csv. You might see a few miscellaneous errors for certain tickers (e.g 'Exceeded 30 redirects.'), but this is to be expected.

Stock prediction
Now that we have the training data and the current data, we can finally generate actual predictions. This part of the project is very simple: the only thing you have to decide is the value of the OUTPERFORMANCE parameter (the percentage by which a stock has to beat the S&P500 to be considered a 'buy'). I have set it to 10 by default, but it can easily be modified by changing the variable at the top of the file. Go ahead and run the script:

python stock_prediction.py
You should get something like this:

21 stocks predicted to outperform the S&P500 by more than 10%:
NOC FL SWK NFX LH NSC SCHL KSU DDS GWW AIZ ORLY R SFLY SHW GME DLX DIS AMP BBBY APD
Unit testing
I have included a number of unit tests (in the tests/ folder) which serve to check that things are working properly. However, due to the nature of the some of this projects functionality (downloading big datasets), you will have to run all the code once before running the tests. Otherwise, the tests themselves would have to download huge datasets (which I don't think is optimal).

I thus recommend that you run the tests after you have run all the other scripts (except, perhaps, stock_prediction.py).

To run the tests, simply enter the following into a terminal instance in the project directory:

pytest -v
Please note that it is not considered best practice to include an __init__.py file in the tests/ directory (see here for more), but I have done it anyway because it is uncomplicated and functional.

Where to go from here
I have stated that this project is extensible, so here are some ideas to get you started and possibly increase returns (no promises).

Data acquisition
My personal belief is that better quality data is THE factor that will ultimately determine your performance. Here are some ideas:

Explore the other subfolders in Sentdex's intraQuarter.zip.
Parse the annual reports that all companies submit to the SEC (have a look at the Edgar Database)
Try to find websites from which you can scrape fundamental data (this has been my solution).
Ditch US stocks and go global – perhaps better results may be found in markets that are less-liquid. It'd be interesting to see whether the predictive power of features vary based on geography.
Buy Quandl data, or experiment with alternative data.
Data preprocessing
Build a more robust parser using BeautifulSoup
In this project, I have just ignored any rows with missing data, but this reduces the size of the dataset considerably. Are there any ways you can fill in some of this data?
hint: if the PE ratio is missing but you know the stock price and the earnings/share...
hint 2: how different is Apple's book value in March to its book value in June?
Some form of feature engineering
e.g, calculate Graham's number and use it as a feature
some of the features are probably redundant. Why not remove them to speed up training?
Speed up the construction of keystats.csv.
hint: don't keep appending to one growing dataframe! Split it into chunks
Machine learning
Altering the machine learning stuff is probably the easiest and most fun to do.

The most important thing if you're serious about results is to find the problem with the current backtesting setup and fix it. This will likely be quite a sobering experience, but if your backtest is done right, it should mean that any observed outperformance on your test set can be traded on (again, do so at your own discretion).
Try a different classifier – there is plenty of research that advocates the use of SVMs, for example. Don't forget that other classifiers may require feature scaling etc.
Hyperparameter tuning: use gridsearch to find the optimal hyperparameters for your classifier. But make sure you don't overfit!
Make it deep – experiment with neural networks (an easy way to start is with sklearn.neural_network).
Change the classification problem into a regression one: will we achieve better results if we try to predict the stock return rather than whether it outperformed?
Run the prediction multiple times (perhaps using different hyperparameters?) and select the k most common stocks to invest in. This is especially important if the algorithm is not deterministic (as is the case for Random Forest)
Experiment with different values of the outperformance parameter.
Should we really be trying to predict raw returns? What happens if a stock achieves a 20% return but does so by being highly volatile?
Try to plot the importance of different features to 'see what the machine sees'.
Contributing
Feel free to fork, play around, and submit PRs. I would be very grateful for any bug fixes or more unit tests.

This project was originally based on Sentdex's excellent machine learning tutorial, but it has since evolved far beyond that and the code is almost completely different. The complete series is also on his website.
