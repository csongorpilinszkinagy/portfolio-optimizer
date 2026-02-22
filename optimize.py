import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage
from pypfopt import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
import matplotlib.pyplot as plt

us_expect = pd.read_csv("data/us_expect.csv")
us_tickers = us_expect["ticker"].to_list()
us_expect_rets = us_expect["return"].to_list()

hu_expect = pd.read_csv("data/hu_expect.csv")
hu_tickers = hu_expect["ticker"].to_list()
hu_expect_rets = hu_expect["return"].to_list()

tickers = us_tickers + hu_tickers
expect_rets = us_expect_rets + hu_expect_rets

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
covariances = CovarianceShrinkage(prices).ledoit_wolf()

mu = expect_rets
S = covariances

ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plot_efficient_frontier(ef, ax=ax, show_assets=False)

n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

ef = EfficientFrontier(mu, S)
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
weights = ef.clean_weights()

for ticker, weight in weights.items():
    print(ticker, round(weight, 2))

ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("pics/efficient_frontier.png")