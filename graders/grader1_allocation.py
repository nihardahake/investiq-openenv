# graders/grader1_allocation.py

def grade(action: dict, user: dict) -> float:
    """
    Grades how well the agent allocated equity/debt/gold
    based on the user's risk score.
    Score: 0.0 to 1.0
    """
    score      = 0.0
    risk       = user["risk_score"]
    equity     = action["equity_pct"]
    debt       = action["debt_pct"]
    gold       = action["gold_pct"]
    total      = equity + debt + gold

    # ── Rule 1: Allocation matches risk profile (50%) ──
    if risk < 40:          # conservative
        if equity <= 30:       score += 0.25
        if debt   >= 50:       score += 0.25
    elif risk < 70:        # balanced
        if 35 <= equity <= 65: score += 0.25
        if 20 <= debt   <= 40: score += 0.25
    else:                  # aggressive
        if equity >= 60:       score += 0.25
        if debt   <= 20:       score += 0.25

    # ── Rule 2: Allocations sum to ~100% (30%) ──
    if 97 <= total <= 103:
        score += 0.30

    # ── Rule 3: Gold is reasonable (20%) ──
    if 5 <= gold <= 20:
        score += 0.20

    return round(min(score, 1.0), 4)