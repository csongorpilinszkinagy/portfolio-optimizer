import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage
from pypfopt import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
import matplotlib.pyplot as plt



expectations = pd.read_csv("data/expectations.csv")
tickers = expectations["ticker"].to_list()
expected_returns = expectations["CAGR"].to_list()

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
covariances = CovarianceShrinkage(prices).ledoit_wolf()

mu = expected_returns
S = covariances

ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plot_efficient_frontier(ef, ax=ax, show_assets=False)

ef = EfficientFrontier(expected_returns, covariances)
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
weights = ef.clean_weights()

for ticker, weight in weights.items():
    print(ticker, weight)

ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("pics/efficient_frontier.png")