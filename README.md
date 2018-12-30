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
