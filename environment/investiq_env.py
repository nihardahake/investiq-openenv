# environment/investiq_env.py
from graders.grader1_allocation import grade as grade1
from graders.grader2_stocks import grade as grade2
from graders.grader3_portfolio import grade as grade3
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from utils.data_fetcher import fetch_all_stocks
from utils.feature_engine import build_feature_dataframe
from utils.stock_universe import EQUITY_STOCKS, DEBT_STOCKS, GOLD_STOCKS, SECTOR_MAP
from services.risk_calculator import calculate_risk_score
from services.allocation import get_allocation

# ─────────────────────────────────────────
# Typed Models — required by OpenEnv spec
# ─────────────────────────────────────────

class UserProfile(BaseModel):
    risk_score:        int
    investment_amount: float
    income:            str
    risk_appetite:     str
    time_horizon:      str
    goal:              str

class MarketSnapshot(BaseModel):
    ticker:        str
    sharpe:        float
    momentum:      float
    annual_return: float
    volatility:    float
    sector:        str

class EnvironmentState(BaseModel):
    task_id:           str
    step:              int
    done:              bool
    user:              UserProfile
    available_stocks:  list[str]
    market_data:       list[MarketSnapshot]
    current_portfolio: dict
    score_so_far:      float

class PortfolioAction(BaseModel):
    equity_pct:      float   # 0–100
    debt_pct:        float   # 0–100
    gold_pct:        float   # 0–100
    selected_stocks: list[str]

class StepResult(BaseModel):
    state:   EnvironmentState
    reward:  float
    done:    bool
    info:    dict


# ─────────────────────────────────────────
# Random profile generator
# ─────────────────────────────────────────

INCOME_OPTIONS    = ["under-5", "5-10", "10-20", "20-50", "50+"]
HORIZON_OPTIONS   = ["under-1", "1-3", "3-5", "5-10", "10+"]
GOAL_OPTIONS      = ["emergency-fund", "short-term", "retirement", "wealth-creation"]
APPETITE_OPTIONS  = ["low", "moderate", "high"]

def generate_random_profile() -> UserProfile:
    income       = np.random.choice(INCOME_OPTIONS)
    appetite     = np.random.choice(APPETITE_OPTIONS)
    horizon      = np.random.choice(HORIZON_OPTIONS)
    goal         = np.random.choice(GOAL_OPTIONS)
    amount       = float(np.random.choice([50000, 100000, 200000, 500000, 1000000]))

    risk_score   = calculate_risk_score(income, appetite, horizon, goal)

    return UserProfile(
        risk_score        = risk_score,
        investment_amount = amount,
        income            = income,
        risk_appetite     = appetite,
        time_horizon      = horizon,
        goal              = goal,
    )


# ─────────────────────────────────────────
# Main Environment Class
# ─────────────────────────────────────────

class InvestIQEnv:

    def __init__(self, task_id: str = "task1_allocation"):
        self.task_id         = task_id
        self.current_state   = None
        self.step_count      = 0
        self.max_steps       = self._get_max_steps(task_id)
        self.score_so_far    = 0.0
        self.features_df     = None
        self.available_stocks = []

    def _ensure_market_data(self):
   
     if self.features_df is None:
        print("Loading market data (first request)...")
        self._load_market_data()
        print(f"Market data ready — {len(self.available_stocks)} stocks")   

    def _get_max_steps(self, task_id: str) -> int:
        return {"task1_allocation": 1,
                "task2_stock_selection": 2,
                "task3_full_portfolio": 3}.get(task_id, 1)

    def _load_market_data(self):
        all_data          = fetch_all_stocks()
        self.features_df  = build_feature_dataframe(all_data)
        self.available_stocks = [
            t for t in EQUITY_STOCKS
            if t in self.features_df.index
        ]

    def _build_market_snapshots(self) -> list[MarketSnapshot]:
        snapshots = []
        for ticker in self.available_stocks:
            row = self.features_df.loc[ticker]
            snapshots.append(MarketSnapshot(
                ticker        = ticker,
                sharpe        = round(float(row["sharpe"]), 4),
                momentum      = round(float(row["momentum"]), 4),
                annual_return = round(float(row["annual_return"]), 4),
                volatility    = round(float(row["volatility"]), 4),
                sector        = str(row.get("sector", "other")),
            ))
        return snapshots

    # ─────────────────────────────────────
    # reset() — start fresh episode
    # ─────────────────────────────────────
    def reset(self) -> EnvironmentState:
        self._ensure_market_data()
        self.step_count   = 0
        self.score_so_far = 0.0
        user              = generate_random_profile()

        self.current_state = EnvironmentState(
            task_id           = self.task_id,
            step              = 0,
            done              = False,
            user              = user,
            available_stocks  = self.available_stocks,
            market_data       = self._build_market_snapshots(),
            current_portfolio = {
                "equity_pct":      0.0,
                "debt_pct":        0.0,
                "gold_pct":        0.0,
                "selected_stocks": [],
            },
            score_so_far = 0.0,
        )
        return self.current_state

    # ─────────────────────────────────────
    # step() — agent takes action
    # ─────────────────────────────────────
    def step(self, action: PortfolioAction) -> StepResult:
        if self.current_state is None:
            raise ValueError("Call reset() before step()")

        self.step_count += 1

        # Validate action
        action = self._validate_action(action)

        # Compute reward for this step
        reward = self._compute_reward(action)
        self.score_so_far += reward

        # Update portfolio
        self.current_state.current_portfolio = {
            "equity_pct":      action.equity_pct,
            "debt_pct":        action.debt_pct,
            "gold_pct":        action.gold_pct,
            "selected_stocks": action.selected_stocks,
        }

        done = self.step_count >= self.max_steps
        self.current_state.step         = self.step_count
        self.current_state.done         = done
        self.current_state.score_so_far = round(self.score_so_far / self.step_count, 4)

        return StepResult(
            state  = self.current_state,
            reward = round(reward, 4),
            done   = done,
            info   = {
                "step":        self.step_count,
                "max_steps":   self.max_steps,
                "task_id":     self.task_id,
                "equity_pct":  action.equity_pct,
                "debt_pct":    action.debt_pct,
                "gold_pct":    action.gold_pct,
                "stocks":      action.selected_stocks,
            }
        )

    # ─────────────────────────────────────
    # state() — return current state
    # ─────────────────────────────────────
    def get_state(self) -> EnvironmentState:
        if self.current_state is None:
            raise ValueError("Call reset() first")
        return self.current_state

    # ─────────────────────────────────────
    # Validate action inputs
    # ─────────────────────────────────────
    def _validate_action(self, action: PortfolioAction) -> PortfolioAction:
        # Clamp percentages
        equity = max(0.0, min(100.0, action.equity_pct))
        debt   = max(0.0, min(100.0, action.debt_pct))
        gold   = max(0.0, min(100.0, action.gold_pct))

        # Normalize to sum to 100
        total = equity + debt + gold
        if total > 0:
            equity = round(equity / total * 100, 2)
            debt   = round(debt   / total * 100, 2)
            gold   = round(100 - equity - debt, 2)

        # Validate stock tickers
        valid_stocks = [
            s for s in action.selected_stocks
            if s in self.available_stocks
        ][:4]   # max 4 stocks

        return PortfolioAction(
            equity_pct      = equity,
            debt_pct        = debt,
            gold_pct        = gold,
            selected_stocks = valid_stocks,
        )

    # ─────────────────────────────────────
    # Reward function
    # ─────────────────────────────────────
    def _compute_reward(self, action: PortfolioAction) -> float:
        action_dict = {
            "equity_pct": action.equity_pct,
            "debt_pct": action.debt_pct,
            "gold_pct": action.gold_pct,
            "selected_stocks": action.selected_stocks,
        }

        user_dict = self.current_state.user.dict()
        market = [m.dict() for m in self.current_state.market_data]

        if self.task_id == "task1_allocation":
            return grade1(action_dict, user_dict)

        elif self.task_id == "task2_stock_selection":
            return grade2(action_dict, user_dict, market)

        elif self.task_id == "task3_full_portfolio":
            return grade3(action_dict, user_dict, market, self.step_count)

        # ✅ ALWAYS LAST
        return 0.0