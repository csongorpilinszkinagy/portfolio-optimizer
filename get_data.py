import pandas as pd
import yfinance as yf

us_expect = pd.read_csv("data/us_expect.csv")
hu_expect = pd.read_csv("data/hu_expect.csv")

us_tickers = us_expect["ticker"].to_list()
hu_tickers = hu_expect["ticker"].to_list()

us_prices = yf.download(us_tickers, start="2015-01-01")["Close"]
hu_prices = yf.download(hu_tickers, start="2015-01-01")["Close"]

fx = yf.download("HUFUSD=X", start="2015-01-01")["Close"]
hu_prices = pd.concat([hu_prices, fx], axis=1, sort=False)
hu_prices["HUFUSD=X"] = hu_prices["HUFUSD=X"].ffill()

for ticker in hu_tickers:
    hu_prices[ticker] = hu_prices[ticker] * hu_prices["HUFUSD=X"]

prices = pd.concat([us_prices, hu_prices[hu_tickers]], axis=1, sort=False).sort_index()
prices.to_csv("data/prices.csv")