import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage
from pypfopt import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier, plot_covariance
import matplotlib.pyplot as plt

risk_free_rate = 0.04

us_expect = pd.read_csv("data/exp_returns_us.csv")
us_expect_rets = us_expect["return"].to_list()

hu_expect = pd.read_csv("data/exp_returns_hu.csv")
hu_expect_rets = hu_expect["return"].to_list()

expect_rets = us_expect_rets + hu_expect_rets

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
covariances = CovarianceShrinkage(prices).ledoit_wolf()
plot_covariance(covariances)
plt.savefig(f"pics/covariance_matrix.png")

mu = expect_rets
S = covariances
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plot_efficient_frontier(ef, ax=ax, show_assets=True)

n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

ef = EfficientFrontier(mu, S)
ef.max_sharpe(risk_free_rate)
ret_tangent, std_tangent, _ = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=True)
sharpe_ratio = (ret_tangent - risk_free_rate) / std_tangent
vols = np.linspace(0, std_tangent*2, 100)
cml_returns = risk_free_rate + sharpe_ratio * vols

sharpe_weights = ef.clean_weights(rounding=2)
df = pd.DataFrame(list(sharpe_weights.items()), columns=["ticker", "weight"])
df.to_csv(f"data/weights_sharpe.csv", index=False)

margin_return = 0.6

leverage = (margin_return - risk_free_rate) / (ret_tangent - risk_free_rate)
margin_vol = round(leverage * std_tangent, 2)
print(f"Expected annual return on margin: {margin_return*100}%")
print(f"Annual volatility on margin: {margin_vol*100}%")
print(f"Leverage: {round(leverage, 2)}")
margin_weights = {k: round(v * leverage, 2) for k, v in sharpe_weights.items()}
df = pd.DataFrame(list(margin_weights.items()), columns=["ticker", "weight"])
df.to_csv(f"data/weights_margin.csv", index=False)


ax.scatter(std_tangent, ret_tangent, marker="*", s=100, color="red", label="Max Sharpe", zorder=2)
ax.scatter(margin_vol, margin_return, marker="o", s=100, color="green", label="Margin return", zorder=2)
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r", zorder=-1)
ax.plot(vols, cml_returns, linestyle="--", color="orange", label="CML", zorder=-1)
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig(f"pics/efficient_frontier.png")