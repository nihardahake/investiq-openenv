# utils/feature_engine.py

import pandas as pd
import numpy as np
from utils.data_fetcher import fetch_all_stocks, get_daily_returns, clean_stock_data
from utils.stock_universe import SECTOR_MAP

TRADING_DAYS = 252


def to_scalar(value):
    """
    Forces any pandas Series, DataFrame, or numpy array
    down to a single Python float. Kills the ambiguity error.
    """
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return float(value.squeeze().iloc[0] if len(value) > 1 else value.squeeze())
    if isinstance(value, np.ndarray):
        return float(value.flat[0])
    return float(value)


def compute_annual_return(returns: pd.Series) -> float:
    avg_daily = to_scalar(returns.mean())
    annual = (1 + avg_daily) ** TRADING_DAYS - 1
    return round(annual * 100, 4)


def compute_volatility(returns: pd.Series) -> float:
    std = to_scalar(returns.std())
    vol = std * np.sqrt(TRADING_DAYS)
    return round(vol * 100, 4)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 6.0) -> float:
    annual_return = compute_annual_return(returns)
    volatility = compute_volatility(returns)
    if volatility == 0:
        return 0.0
    sharpe = (annual_return - risk_free_rate) / volatility
    return round(float(sharpe), 4)


def compute_momentum(data: pd.DataFrame) -> float:
    clean = clean_stock_data(data)

    # Force Close to a plain 1D Series
    close = clean["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    if len(close) < TRADING_DAYS:
        return 0.0

    recent = close.iloc[-TRADING_DAYS:]
    start  = float(recent.iloc[0])
    end    = float(recent.iloc[-1])

    if start == 0:
        return 0.0

    return round(((end - start) / start) * 100, 4)


def compute_max_drawdown(returns: pd.Series) -> float:
    # Force to 1D Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()

    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = to_scalar(drawdown.min())
    return round(max_dd * 100, 4)


def compute_features_for_stock(ticker: str, data: pd.DataFrame) -> dict | None:
    try:
        returns = get_daily_returns(data)

        if returns is None or returns.empty or len(returns) < 100:
            print(f"  Skipping {ticker} — insufficient returns data")
            return None

        # Force to plain 1D float Series
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        returns = returns.astype(float)

        return {
            "ticker":        ticker,
            "sector":        SECTOR_MAP.get(ticker, "other"),
            "annual_return": compute_annual_return(returns),
            "volatility":    compute_volatility(returns),
            "sharpe":        compute_sharpe(returns),
            "momentum":      compute_momentum(data),
            "max_drawdown":  compute_max_drawdown(returns),
        }

    except Exception as e:
        print(f"  Error on {ticker}: {e}")
        return None


def build_feature_dataframe(all_data: dict) -> pd.DataFrame:
    rows = []

    for ticker, data in all_data.items():
        print(f"  Computing features for {ticker}...")
        features = compute_features_for_stock(ticker, data)
        if features is not None:
            rows.append(features)

    if not rows:
        raise ValueError("No features computed — ALL stocks failed")

    df = pd.DataFrame(rows)
    df = df.set_index("ticker")
    return df