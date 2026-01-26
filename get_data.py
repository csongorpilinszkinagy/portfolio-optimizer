import pandas as pd
import yfinance as yf

expectations = pd.read_csv("data/expectations.csv")

tickers = expectations['ticker'].to_list()

prices = yf.download(tickers, start="2015-01-01")["Close"]
prices = prices.dropna(axis=1, how='all')     # Drop bad tickers entirely
prices = prices.dropna(axis=0, how='any')     # Drop rows with missing prices

prices.to_csv("data/prices.csv")