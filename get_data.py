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
hu_prices = hu_prices[hu_tickers]

us_prices = us_prices.dropna(how="all")
hu_prices = hu_prices.dropna(how="all")

us_prices.to_csv("data/us_prices.csv")
hu_prices.to_csv("data/hu_prices.csv")

all_prices = pd.concat([us_prices, hu_prices], axis=1, sort=False).sort_index()
all_prices.to_csv("data/all_prices.csv")