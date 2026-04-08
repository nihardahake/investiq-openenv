# graders/grader3_portfolio.py

import numpy as np
from utils.stock_universe import SECTOR_MAP

def grade(action: dict, user: dict,
          market_data: list, step: int) -> float:
    """
    Grades the full portfolio over 3 steps.
    Each step has a different focus:
      Step 1 → allocation quality
      Step 2 → stock selection quality
      Step 3 → rebalancing / final portfolio quality
    Score: 0.0 to 1.0
    """

    market = {m["ticker"]: m for m in market_data}
    risk   = user["risk_score"]
    stocks = action.get("selected_stocks", [])
    equity = action["equity_pct"]
    debt   = action["debt_pct"]
    gold   = action["gold_pct"]

    if step == 1:
        return _grade_step1_allocation(equity, debt, gold, risk)

    elif step == 2:
        return _grade_step2_selection(stocks, market, risk)

    elif step == 3:
        return _grade_step3_rebalance(
            equity, debt, gold, stocks, market, risk
        )

    return 0.0


def _grade_step1_allocation(equity, debt, gold, risk) -> float:
    """Step 1: Did the agent allocate correctly for this risk level?"""
    score = 0.0
    total = equity + debt + gold

    if risk < 40:
        if equity <= 30: score += 0.35
        if debt   >= 50: score += 0.35
    elif risk < 70:
        if 35 <= equity <= 65: score += 0.35
        if 20 <= debt   <= 40: score += 0.35
    else:
        if equity >= 60: score += 0.35
        if debt   <= 20: score += 0.35

    if 97 <= total <= 103:
        score += 0.30

    return round(min(score, 1.0), 4)


def _grade_step2_selection(stocks, market, risk) -> float:
    """Step 2: Did the agent pick quality stocks?"""
    if not stocks:
        return 0.0

    score = 0.0

    # Sharpe quality
    sharpes = [market[t]["sharpe"] for t in stocks if t in market]
    if sharpes:
        avg_sharpe = np.mean(sharpes)
        score += min(avg_sharpe / 1.5, 1.0) * 0.40

    # Sector diversity
    sectors = [SECTOR_MAP.get(t, "other") for t in stocks]
    unique  = len(set(sectors))
    score  += min(unique / 4, 1.0) * 0.30

    # Momentum — is the stock trending up?
    momentums = [market[t]["momentum"] for t in stocks if t in market]
    if momentums:
        avg_momentum = np.mean(momentums)
        if avg_momentum > 10:
            score += 0.20
        elif avg_momentum > 0:
            score += 0.10

    # Picked enough stocks
    if len(stocks) >= 3:
        score += 0.10

    return round(min(score, 1.0), 4)


def _grade_step3_rebalance(equity, debt, gold,
                            stocks, market, risk) -> float:
    """
    Step 3: Full portfolio quality check.
    This is the hardest — we check everything together.
    """
    score = 0.0

    # ── Allocation still correct (25%) ──
    if risk < 40:
        if equity <= 35: score += 0.25
    elif risk < 70:
        if 30 <= equity <= 70: score += 0.25
    else:
        if equity >= 55: score += 0.25

    # ── Portfolio Sharpe (25%) ──
    sharpes = [market[t]["sharpe"] for t in stocks if t in market]
    if sharpes:
        avg_sharpe = np.mean(sharpes)
        score += min(avg_sharpe / 2.0, 0.25)

    # ── Diversification (25%) ──
    sectors = [SECTOR_MAP.get(t, "other") for t in stocks]
    unique  = len(set(sectors))
    score  += min(unique / 4, 1.0) * 0.25

    # ── Risk-adjusted momentum (15%) ──
    momentums = [market[t]["momentum"] for t in stocks if t in market]
    if momentums:
        avg_momentum = np.mean(momentums)
        if risk < 40 and avg_momentum > 5:
            score += 0.15
        elif risk < 70 and avg_momentum > 10:
            score += 0.15
        elif risk >= 70 and avg_momentum > 15:
            score += 0.15

    # ── Constraint check (10%) ──
    total = equity + debt + gold
    if 97 <= total <= 103:
        score += 0.10

    return round(min(score, 1.0), 4)