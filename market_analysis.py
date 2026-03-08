import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

windows = {
    "1y": 1,
    "2y": 2,
    "5y": 5,
    "10y": 10
}

return_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

start_date = "1996-01-01"
end_date = "2025-12-31"

constituents_url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
const = pd.read_csv(constituents_url)
const["start_date"] = pd.to_datetime(const["start_date"])
const["end_date"] = pd.to_datetime(const["end_date"])
const["end_date"] = const["end_date"].fillna(pd.Timestamp.today().normalize())

tickers = const["ticker"].unique().tolist()

print("Downloading price data (this may take a few minutes)...")
prices = yf.download(tickers, start=start_date, end=end_date)["Close"]
monthly = prices.resample("ME").mean()

returns = {}
for name, window in windows.items():
    returns[name] = (monthly / monthly.shift(window*12)) ** (1/window) - 1

dates = monthly.index
membership = pd.DataFrame(False, index=dates, columns=tickers)

for _, row in const.iterrows():
    ticker = row["ticker"]
    start = row["start_date"]
    end = row["end_date"]
    membership.loc[
        (membership.index >= start) &
        (membership.index <= end),
        ticker
    ] = True

results = {}

for name, window in windows.items():
    r = returns[name]
    # csak akkor számolunk, ha a cég az egész ablak alatt tag volt
    valid_mask = membership & membership.shift(window*12)
    r_valid = r.where(valid_mask)

    stats = []
    for date in r_valid.index:
        row = r_valid.loc[date].dropna()
        if len(row) == 0:
            continue
        total = len(row)
        res = {"date": date, "n": total}
        for th in return_thresholds:
            res[f">{int(th*100)}%"] = (row > th).sum() / total
        stats.append(res)

    results[name] = pd.DataFrame(stats).set_index("date")

for name in results:
    print(f"\n==== {name} rolling window ====\n")
    print(results[name].tail(10))

os.makedirs("pics", exist_ok=True)

for window_name, df in results.items():
    plt.figure(figsize=(12,6))
    
    for th in return_thresholds:
        plt.plot(df.index, df[f">{int(th*100)}%"], label=f">{int(th*100)}%")
    
    plt.title(f"{window_name} rolling window: % of stocks above return thresholds")
    plt.xlabel("Date")
    plt.ylabel("Fraction of stocks")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"pics/{window_name}_window.png", dpi=300, bbox_inches='tight')
    plt.close()