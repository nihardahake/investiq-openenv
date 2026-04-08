# utils/data_fetcher.py

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import date
from utils.stock_universe import EQUITY_STOCKS, DEBT_STOCKS, GOLD_STOCKS

CACHE_FILE = "cache/stock_data.pkl"
CACHE_DATE = "cache/last_fetch.txt"


def is_cache_fresh() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    if not os.path.exists(CACHE_DATE):
        return False
    with open(CACHE_DATE, "r") as f:
        last = f.read().strip()
    return last == str(date.today())


def save_cache(data: dict):
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    with open(CACHE_DATE, "w") as f:
        f.write(str(date.today()))
    print("Cache saved.")


def load_cache() -> dict:
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    try:
        data = yf.download(
            ticker, period=period,
            interval="1d", progress=False
        )

        if data is None or data.empty:
            print(f"  Skipping {ticker} — empty data")
            return None

        data = flatten_columns(data)

        if "Close" not in data.columns:
            print(f"  Skipping {ticker} — no Close column")
            return None

        if len(data) < 200:
            print(f"  Skipping {ticker} — not enough history ({len(data)} days)")
            return None

        return data

    except Exception as e:
        print(f"  Failed {ticker}: {e}")
        return None


def fetch_all_stocks() -> dict:
    # ── Return cache if already fetched today ──
    if is_cache_fresh():
        print("Using cached stock data (fetched today already)")
        return load_cache()

    # ── Otherwise fetch fresh ──
    all_data   = {}
    all_tickers = EQUITY_STOCKS + DEBT_STOCKS + GOLD_STOCKS

    print(f"Fetching fresh data for {len(all_tickers)} stocks...")

    for ticker in all_tickers:
        print(f"  Fetching {ticker}...")
        data = fetch_stock_data(ticker)
        if data is not None:
            all_data[ticker] = data

    print(f"\nFetched {len(all_data)} / {len(all_tickers)} stocks")

    save_cache(all_data)
    return all_data


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)
    result = df[["Close"]].copy()
    result = result.ffill()
    result = result.dropna()
    return result


def get_daily_returns(data: pd.DataFrame) -> pd.Series:
    if data is None or data.empty:
        return pd.Series()

    clean = clean_stock_data(data)

    if clean is None or clean.empty:
        return pd.Series()

    close = clean["Close"].squeeze()

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close   = close.astype(float)
    returns = close.pct_change().dropna()
    return returns