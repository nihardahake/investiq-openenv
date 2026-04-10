# inference.py

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# Config — required by hackathon spec
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:8000")

client = OpenAI(
    api_key  = os.getenv("GROQ_API_KEY") or HF_TOKEN,
    base_url = API_BASE_URL,
)

# ─────────────────────────────────────────
# Helper — call environment endpoints
# ─────────────────────────────────────────

def reset(task_id: str) -> dict:
    res = requests.post(f"{ENV_URL}/reset?task_id={task_id}")
    res.raise_for_status()
    return res.json()

def step(task_id: str, action: dict) -> dict:
    res = requests.post(
        f"{ENV_URL}/step?task_id={task_id}",
        json    = action,
        headers = {"Content-Type": "application/json"}
    )
    res.raise_for_status()
    return res.json()

# ─────────────────────────────────────────
# Agent — uses LLM to decide action
# ─────────────────────────────────────────

def agent_decide(state: dict, task_id: str) -> dict:
    """
    Sends state to LLM, gets back a portfolio action.
    """

    user         = state["user"]
    risk_score   = user["risk_score"]
    risk_level   = "conservative" if risk_score < 40 else \
                   "balanced"     if risk_score < 70 else \
                   "aggressive"

    # Build top stocks summary for the prompt
    market_data  = state.get("market_data", [])
    top_stocks   = sorted(
        market_data,
        key     = lambda x: x["sharpe"],
        reverse = True
    )[:10]

    stock_summary = "\n".join([
        f"  {s['ticker']}: sharpe={s['sharpe']}, "
        f"momentum={s['momentum']:.1f}%, "
        f"sector={s['sector']}"
        for s in top_stocks
    ])

    prompt = f"""You are an expert Indian investment advisor.

User profile:
- Risk score: {risk_score}/100 ({risk_level} investor)
- Investment amount: ₹{user['investment_amount']:,.0f}
- Income: {user['income']} LPA
- Goal: {user['goal']}
- Time horizon: {user['time_horizon']} years

Task: {task_id}
Current step: {state['step']}

Top 10 stocks by Sharpe ratio:
{stock_summary}

Based on this profile, decide a portfolio allocation.

Rules:
- equity_pct + debt_pct + gold_pct MUST sum to exactly 100
- For conservative (risk < 40): equity <= 30, debt >= 50
- For balanced (risk 40-70): equity 40-60, debt 20-40
- For aggressive (risk > 70): equity >= 60, debt <= 20
- Select exactly 4 stocks from the top stocks list above
- Pick stocks from different sectors for diversification

Respond ONLY with valid JSON in this exact format:
{{
  "equity_pct": <number>,
  "debt_pct": <number>,
  "gold_pct": <number>,
  "selected_stocks": ["TICKER1.NS", "TICKER2.NS", "TICKER3.NS", "TICKER4.NS"]
}}

No explanation. JSON only."""

    response = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 200,
        temperature = 0.1,    # low temp = more consistent outputs
    )

    raw = response.choices[0].message.content.strip()

    # Clean up response — remove markdown if LLM adds it
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Warning: LLM returned invalid JSON, using fallback action")
        action = _fallback_action(risk_score, market_data)

    # Validate and clean
    action = _validate_action(action, market_data)
    return action


def _fallback_action(risk_score: int, market_data: list) -> dict:
    """
    Used if LLM returns invalid JSON.
    Simple rule-based fallback.
    """
    if risk_score < 40:
        equity, debt, gold = 25, 65, 10
    elif risk_score < 70:
        equity, debt, gold = 55, 35, 10
    else:
        equity, debt, gold = 70, 20, 10

    # Pick top 4 by Sharpe
    top4 = sorted(market_data, key=lambda x: x["sharpe"], reverse=True)[:4]

    return {
        "equity_pct":      equity,
        "debt_pct":        debt,
        "gold_pct":        gold,
        "selected_stocks": [s["ticker"] for s in top4],
    }


def _validate_action(action: dict, market_data: list) -> dict:
    """
    Makes sure action is valid before sending to environment.
    """
    available = {m["ticker"] for m in market_data}

    equity = float(action.get("equity_pct", 60))
    debt   = float(action.get("debt_pct",   30))
    gold   = float(action.get("gold_pct",   10))

    # Normalize to sum to 100
    total = equity + debt + gold
    if total > 0:
        equity = round(equity / total * 100, 1)
        debt   = round(debt   / total * 100, 1)
        gold   = round(100 - equity - debt,  1)

    # Validate stocks
    raw_stocks = action.get("selected_stocks", [])
    stocks = [s for s in raw_stocks if s in available]

    # If not enough valid stocks, fill from top Sharpe
    if len(stocks) < 4:
        top = sorted(market_data, key=lambda x: x["sharpe"], reverse=True)
        for s in top:
            if s["ticker"] not in stocks:
                stocks.append(s["ticker"])
            if len(stocks) == 4:
                break

    return {
        "equity_pct":      equity,
        "debt_pct":        debt,
        "gold_pct":        gold,
        "selected_stocks": stocks[:4],
    }


# ─────────────────────────────────────────
# Run one task — reset → step(s) → score
# ─────────────────────────────────────────

def run_task(task_id: str, max_steps: int) -> float:

    # ── REQUIRED: START block ──────────────────────
    print(f"[START] task={task_id}", flush=True)

    try:
        state  = reset(task_id)
        print(f"User: risk_score={state['user']['risk_score']}, "
              f"amount=₹{state['user']['investment_amount']:,.0f}",
              flush=True)

        final_score = 0.0

        for step_num in range(1, max_steps + 1):

            # Agent decides
            action = agent_decide(state, task_id)

            # Take step
            result      = step(task_id, action)
            reward      = result["reward"]
            done        = result["done"]
            final_score = result["state"]["score_so_far"]

            # ── REQUIRED: STEP block ───────────────
            print(f"[STEP] step={step_num} reward={reward}", flush=True)

            state = result["state"]

            if done:
                break

    except Exception as e:
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"Error: {e}", flush=True)
        final_score = 0.0

    # ── REQUIRED: END block ────────────────────────
    print(f"[END] task={task_id} score={final_score} steps={max_steps}",
          flush=True)

    return final_score


# ─────────────────────────────────────────
# Main — run all 3 tasks
# ─────────────────────────────────────────

def main():
    print("InvestIQ OpenEnv — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Environment: {ENV_URL}", flush=True)

    tasks = [
        ("task1_allocation",      1),
        ("task2_stock_selection", 2),
        ("task3_full_portfolio",  3),
    ]

    scores = {}

    for task_id, max_steps in tasks:
        score          = run_task(task_id, max_steps)
        scores[task_id] = score

    # Final summary
    print("\n=== FINAL SCORES ===", flush=True)
    for task_id, score in scores.items():
        print(f"{task_id}: {score}", flush=True)

    avg = sum(scores.values()) / len(scores)
    print(f"Average: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()