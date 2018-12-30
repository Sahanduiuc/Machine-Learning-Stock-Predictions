# Machine-Learning-Stock-Predictions
Machine Learning Stock Predictions is designed to be an intuitive and extensible project that harnesses machine learning to make stock predictions. 

Initally, I clean and prepare a dataset of historical stock prices (from 31/12/2017 to 31/12/2018) and fundamental infomration using pandas, after which I apply a scikit-learn classifier to uncover the relationship between stock fundamentals (e.g. PE ratio, debt/equity, float, etc.) and the subsequent annual price change compared with the an index. I then conduct a backtest, before generating predictions on current data.

Note: Machine Learning Stock Predictions predicts which stocks will outperform the S&P500; however, it does not suggest how best to combine stocks into a portfolio. 

Here is a high-level overview of the steps I take:

1. Scrape historical fundamental data from a non-standard resource on the internet (an outdated website with no convenient API)
2. Download historical stock prices from Yahoo, Google, and Quandl
3. Calculate the forward annual returns for each stock and the S&P500 for each time snapshot
4. Combine and clean all of this data into a readable csv, using pandas. ALso account for missing data. 
5. Back test a large number of different machine learning models K-NN, Naive Bayes, Random Forest, SVC, and XGBoost, and limit the effects of data snooping. 
5. Parameter tune the most promising models using scikit-learn’s gridsearch.
6. Manually tune certain parameters to reduce overfitting.
7. Repeatedly train the models on all of the data, and generate predictions.
8. Combine the predictions using classical techniques, such as efficient frontier and the Kelly criterion.
