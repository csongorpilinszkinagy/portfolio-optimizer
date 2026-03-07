import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

with open("secrets/sheet_id", "r", encoding="utf-8") as f:
    sheet_id = f.read().strip()

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("secrets/credentials.json", scope)
client = gspread.authorize(creds)

def update_sheet(sheet_name, file_name):
    spreadsheet = client.open_by_key(sheet_id)
    sheet = spreadsheet.worksheet(sheet_name)

    tickers = sheet.col_values(1)[2:]
    df = pd.read_csv(file_name)

    for i, ticker in enumerate(tickers):
        if ticker:
            value = df.loc[df["ticker"] == ticker, "weight"].values
            if len(value) > 0:
                value = value[0]
                sheet.update_acell(f"C{i+3}", value)
        
update_sheet("Allocation Sharpe", "data/weights_sharpe.csv")
update_sheet("Allocation Margin", "data/weights_margin.csv")