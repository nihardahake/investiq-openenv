# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from environment.investiq_env import (
    InvestIQEnv, PortfolioAction,
    EnvironmentState, StepResult
)

app = FastAPI(
    title       = "InvestIQ OpenEnv",
    description = "Indian stock portfolio management RL environment",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# One environment instance per task
envs = {
    "task1_allocation":      InvestIQEnv("task1_allocation"),
    "task2_stock_selection": InvestIQEnv("task2_stock_selection"),
    "task3_full_portfolio":  InvestIQEnv("task3_full_portfolio"),
}

def get_env(task_id: str) -> InvestIQEnv:
    if task_id not in envs:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown task: {task_id}. Choose from {list(envs.keys())}"
        )
    return envs[task_id]


# ─────────────────────────────────────────
# Required OpenEnv endpoints
# ─────────────────────────────────────────

@app.post("/reset", response_model=EnvironmentState)
def reset(task_id: str = "task1_allocation"):
    """Start a fresh episode."""
    env   = get_env(task_id)
    state = env.reset()
    return state


@app.post("/step", response_model=StepResult)
def step(action: PortfolioAction, task_id: str = "task1_allocation"):
    """Agent takes an action."""
    env    = get_env(task_id)
    result = env.step(action)
    return result


@app.get("/state", response_model=EnvironmentState)
def state(task_id: str = "task1_allocation"):
    """Get current state without taking action."""
    env = get_env(task_id)
    return env.get_state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "id":         "task1_allocation",
                "difficulty": "easy",
                "max_steps":  1,
                "description": "Choose the right equity/debt/gold split for a user"
            },
            {
                "id":         "task2_stock_selection",
                "difficulty": "medium",
                "max_steps":  2,
                "description": "Pick the best NSE stocks for the equity bucket"
            },
            {
                "id":         "task3_full_portfolio",
                "difficulty": "hard",
                "max_steps":  3,
                "description": "Build and rebalance a complete portfolio over 3 steps"
            },
        ]
    }


@app.get("/")
def root():
    return {
        "name":    "InvestIQ OpenEnv",
        "version": "1.0.0",
        "status":  "running",
        "tasks":   list(envs.keys()),
    }