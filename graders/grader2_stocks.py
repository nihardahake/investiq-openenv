# graders/grader2_stocks.py

import numpy as np
from utils.stock_universe import SECTOR_MAP

def grade(action: dict, user: dict, market_data: list) -> float:
    """
    Grades the stock selection quality.
    Looks at Sharpe ratio + sector diversity + risk match.
    Score: 0.0 to 1.0
    """
    score   = 0.0
    stocks  = action.get("selected_stocks", [])
    risk    = user["risk_score"]

    if not stocks:
        return 0.0

    # Build lookup from market_data list
    market = {m["ticker"]: m for m in market_data}

    # ── Rule 1: Stock quality by Sharpe (40%) ──
    sharpes = []
    for ticker in stocks:
        if ticker in market:
            sharpes.append(market[ticker]["sharpe"])

    if sharpes:
        avg_sharpe = np.mean(sharpes)
        # Sharpe > 1.0 is great, > 0.5 is decent
        sharpe_score = min(avg_sharpe / 1.5, 1.0) * 0.40
        score += sharpe_score

    # ── Rule 2: Sector diversification (30%) ──
    sectors        = [SECTOR_MAP.get(t, "other") for t in stocks]
    unique_sectors = len(set(sectors))
    # 4 different sectors = perfect diversification
    diversity_score = min(unique_sectors / 4, 1.0) * 0.30
    score += diversity_score

    # ── Rule 3: Volatility matches risk level (20%) ──
    volatilities = [
        market[t]["volatility"]
        for t in stocks if t in market
    ]

    if volatilities:
        avg_vol = np.mean(volatilities)
        if risk < 40:       # conservative — want low vol
            if avg_vol <= 22: score += 0.20
            elif avg_vol <= 28: score += 0.10
        elif risk < 70:     # balanced — medium vol ok
            if avg_vol <= 32: score += 0.20
            elif avg_vol <= 38: score += 0.10
        else:               # aggressive — any vol ok
            score += 0.20

    # ── Rule 4: Picked enough stocks (10%) ──
    if len(stocks) >= 3:
        score += 0.10

    return round(min(score, 1.0), 4)