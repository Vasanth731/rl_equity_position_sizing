
"""
download_nse_ohlcv.py

Downloads Open, High, Low, Close, Volume (OHLCV) data for ALL NSE-listed
equity stocks and saves them into a single combined CSV (and optionally
one CSV per stock).

HOW TO RUN
----------
1. Install requirements (one time):
       pip install yfinance pandas requests

2. Run the script:
       python download_nse_ohlcv.py

   By default it downloads the last 1 year of daily data for every stock
   in the official NSE equity list. Edit the CONFIG section below to
   change the date range, or to test with just a handful of stocks first.

NOTES
-----
- This script needs an unrestricted internet connection (it will NOT
  work inside a sandboxed/offline environment). Run it on your own PC,
  laptop, or in Google Colab.
- The official NSE list contains ~2,000+ stocks. Downloading full
  history for all of them will take a while (yfinance rate-limits
  requests) and can produce a large file. Start with a small TEST_MODE
  run first to make sure everything works.
- yfinance pulls data from Yahoo Finance, which mirrors NSE prices.
  Indian NSE stock tickers on Yahoo Finance are suffixed with ".NS"
  (e.g. RELIANCE.NS, TCS.NS, VODAFONEIDEA.NS).
"""

import io
import time
import pandas as pd
import requests
import yfinance as yf

# ============================== CONFIG ===============================

# Date range for historical data
START_DATE = "2023-01-01"
END_DATE = None  # None = up to today

# If True, only downloads a small sample (first N stocks) to test the
# script quickly before running the full list. Set to False for the
# complete run across all NSE stocks.
TEST_MODE = False
TEST_MODE_STOCK_COUNT = 10

# ---- SEARCH BY COMPANY NAME ----
# Leave this list empty ([]) to run the normal full/test NSE download.
# Fill it in with one or more company names (or partial names) to ONLY
# download those specific companies instead of the whole exchange.
#
# Examples:
#   COMPANY_NAME_SEARCH = ["Vodafone"]
#   COMPANY_NAME_SEARCH = ["Reliance", "Tata Consultancy", "Infosys"]
#
# Matching is case-insensitive and works on partial names, so "Tata"
# will match every NSE company with "Tata" in its name (Tata Motors,
# Tata Steel, Tata Power, etc.) — check the printed matches before
# confirming, in case a search term pulls in more than you expect.
COMPANY_NAME_SEARCH = []

# Where to save results
COMBINED_OUTPUT_FILE = "nse_all_stocks_ohlcv.csv"
SAVE_INDIVIDUAL_FILES = True   # True = also save one CSV per stock
INDIVIDUAL_FILES_FOLDER = "data2"

# Pause between requests to avoid getting rate-limited by Yahoo Finance
REQUEST_DELAY_SECONDS = 0.5

# =======================================================================


def get_nse_equity_list():
    """
    Fetches the official, up-to-date list of all NSE-listed equity
    symbols directly from the NSE archives.
    Returns a pandas DataFrame with columns like SYMBOL, NAME OF COMPANY, etc.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))
    df.columns = [c.strip() for c in df.columns]
    return df


def find_symbols_by_company_name(nse_list, search_terms):
    """
    Looks up NSE symbols whose company name contains any of the given
    search terms (case-insensitive, partial match).
    Returns a list of matching symbols, printing what matched each term.
    """
    name_col = "NAME OF COMPANY"
    matched_symbols = []

    for term in search_terms:
        matches = nse_list[nse_list[name_col].str.contains(term, case=False, na=False)]

        if matches.empty:
            print(f"  [!] No company name match found for '{term}'")
            continue

        print(f"  Matches for '{term}':")
        for _, row in matches.iterrows():
            print(f"    {row['SYMBOL']}  -  {row[name_col]}")
            matched_symbols.append(row["SYMBOL"])

    # remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in matched_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    return unique_symbols


def download_ohlcv_for_symbol(nse_symbol, start_date, end_date):
    """
    Downloads OHLCV data for a single NSE stock via yfinance.
    nse_symbol should be the plain NSE symbol (e.g. 'RELIANCE'),
    the '.NS' suffix is added automatically.
    """
    yahoo_ticker = f"{nse_symbol}.NS"
    try:
        data = yf.download(
            yahoo_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            return None

        data = data.reset_index()
        data["Symbol"] = nse_symbol
        return data

    except Exception as e:
        print(f"  [!] Failed to download {nse_symbol}: {e}")
        return None


def main():
    import os

    print("Fetching official NSE equity list...")
    nse_list = get_nse_equity_list()
    symbols = nse_list["SYMBOL"].tolist()
    print(f"Found {len(symbols)} NSE-listed equity symbols.")

    if COMPANY_NAME_SEARCH:
        print(f"\nCOMPANY_NAME_SEARCH is set — searching for: {COMPANY_NAME_SEARCH}\n")
        matched = find_symbols_by_company_name(nse_list, COMPANY_NAME_SEARCH)

        if not matched:
            print("\nNo companies matched your search terms. Nothing to download.")
            return

        symbols = matched
        print(f"\nWill download OHLCV data for {len(symbols)} matched symbol(s): {symbols}\n")

    elif TEST_MODE:
        symbols = symbols[:TEST_MODE_STOCK_COUNT]
        print(f"TEST_MODE is ON — only downloading first {len(symbols)} stocks.")
        print("Set TEST_MODE = False in the script to run the full list.\n")

    if SAVE_INDIVIDUAL_FILES and not os.path.exists(INDIVIDUAL_FILES_FOLDER):
        os.makedirs(INDIVIDUAL_FILES_FOLDER)

    all_data = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{total}] Downloading {symbol} ...")
        df = download_ohlcv_for_symbol(symbol, START_DATE, END_DATE)

        if df is not None:
            all_data.append(df)
            if SAVE_INDIVIDUAL_FILES:
                out_path = os.path.join(INDIVIDUAL_FILES_FOLDER, f"{symbol}.csv")
                df.to_csv(out_path, index=False)

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_data:
        print("No data was downloaded. Check your internet connection or symbol list.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(COMBINED_OUTPUT_FILE, index=False)

    print(f"\nDone! Combined OHLCV data for {combined['Symbol'].nunique()} stocks "
          f"saved to: {COMBINED_OUTPUT_FILE}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()