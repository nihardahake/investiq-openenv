def calculate_risk_score(
    income: str,
    risk_appetite: str,
    time_horizon: str,
    goal: str
) -> int:

    income_map = {
        "under-5": 10,
        "5-10": 15,
        "10-20": 20,
        "20-50": 25,
        "50+": 30,
    }

    appetite_map = {
        "low": 10,
        "moderate": 25,
        "high": 35,
    }

    horizon_map = {
        "under-1": 5,
        "1-3": 10,
        "3-5": 15,
        "5-10": 20,
        "10+": 25,
    }

    goal_map = {
        "emergency-fund": 5,
        "short-term": 8,
        "retirement": 12,
        "wealth-creation": 15,
    }

    income_score = income_map.get(income, 15)
    appetite_score = appetite_map.get(risk_appetite, 20)
    horizon_score = horizon_map.get(time_horizon, 15)
    goal_score = goal_map.get(goal, 10)

    raw = income_score + appetite_score + horizon_score + goal_score
    max_possible = 105

    score = round((raw / max_possible) * 100)

    return min(max(score, 0), 100)


def get_strategy_label(score: int) -> str:
    if score < 40:
        return "CONSERVATIVE"
    elif score < 70:
        return "BALANCED"
    else:
        return "AGGRESSIVE"