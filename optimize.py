import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage

expectations = pd.read_csv("data/expectations.csv")
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")

covariances = CovarianceShrinkage(prices).ledoit_wolf()


