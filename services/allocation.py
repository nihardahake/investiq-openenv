# services/allocation.py

ALLOCATION_MAP = {
    "conservative": {
        "equity": 10,
        "mutual_funds": 20,
        "bonds": 50,
        "fixed_deposits": 20,
        "gold": 0,
    },
    "balanced": {
        "equity": 40,
        "mutual_funds": 30,
        "bonds": 20,
        "fixed_deposits": 0,
        "gold": 10,
    },
    "aggressive": {
        "equity": 60,
        "mutual_funds": 20,
        "bonds": 0,
        "fixed_deposits": 0,
        "gold": 10,
        "crypto": 10,
    },
}


def get_allocation(strategy: str) -> dict:
    return ALLOCATION_MAP.get(strategy.lower(), ALLOCATION_MAP["balanced"])


def map_rupee_amounts(allocation: dict, amount: float) -> dict:
    """
    Converts % allocation to actual rupee amounts.
    """
    result = {}
    for key, pct in allocation.items():
        result[f"{key}_pct"] = pct
        result[f"{key}_amt"] = round(amount * pct / 100, 2)
    return result