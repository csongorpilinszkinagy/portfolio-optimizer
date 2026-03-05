import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage
from pypfopt import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier, plot_covariance
import matplotlib.pyplot as plt

us_expect = pd.read_csv("data/us_expect.csv")
us_expect_rets = us_expect["return"].to_list()

hu_expect = pd.read_csv("data/hu_expect.csv")
hu_expect_rets = hu_expect["return"].to_list()

all_expect_rets = us_expect_rets + hu_expect_rets

us_prices = pd.read_csv("data/us_prices.csv", parse_dates=["Date"], index_col="Date")
hu_prices = pd.read_csv("data/hu_prices.csv", parse_dates=["Date"], index_col="Date")
all_prices = pd.read_csv("data/all_prices.csv", parse_dates=["Date"], index_col="Date")


parameters = [
    ("us", us_expect_rets, us_prices),
    ("hu", hu_expect_rets, hu_prices),
    ("all", all_expect_rets, all_prices)
]

for country, expect_rets, prices in parameters:
    covariances = CovarianceShrinkage(prices).ledoit_wolf()
    plot_covariance(covariances)
    plt.savefig(f"pics/{country}_covariance_matrix.png")

    mu = expect_rets
    S = covariances
    ef = EfficientFrontier(mu, S)

    fig, ax = plt.subplots()
    plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
    sharpes = rets / stds

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    ret_tangent, std_tangent, _ = ef.portfolio_performance(verbose=True)
    weights = ef.clean_weights()
    weights = [(ticker, round(weight, 2)) for ticker, weight in weights.items()]

    df = pd.DataFrame(weights, columns=["ticker", "weight"])
    df.to_csv(f"data/{country}_weights.csv", index=False)

    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe", zorder=2)
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r", zorder=-1)
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"pics/{country}_efficient_frontier.png")