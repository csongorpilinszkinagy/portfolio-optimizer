import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

with open("secrets/sheet_id", "r", encoding="utf-8") as f:
    sheet_id = f.read().strip()

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("secrets/credentials.json", scope)
client = gspread.authorize(creds)

spreadsheet = client.open_by_key(sheet_id)
sheet = spreadsheet.worksheet(f"DCF")

data = sheet.get('A:I')
df = pd.DataFrame(data[1:], columns=data[0])
df = df[["Ticker", "CAGR"]]
df.columns = ["ticker", "return"]
df["return"] = df["return"].str.strip().str.replace("%", "")
df["return"] = pd.to_numeric(df["return"], errors="coerce") / 100

us_df = df[~df["ticker"].str.endswith(".BD")]
hu_df = df[df["ticker"].str.endswith(".BD")]

us_df.to_csv(f"data/exp_returns_us.csv", index=False)
hu_df.to_csv(f"data/exp_returns_hu.csv", index=False)
