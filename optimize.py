import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import EfficientFrontier

expectations = pd.read_csv("data/expectations.csv")
tickers = expectations["ticker"].to_list()
expected_returns = expectations["CAGR"].to_list()

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
covariances = CovarianceShrinkage(prices).ledoit_wolf()

ef = EfficientFrontier(expected_returns, covariances)
ef.max_sharpe()
weights = ef.clean_weights()

for ticker, weight in weights.items():
    print(ticker, weight)