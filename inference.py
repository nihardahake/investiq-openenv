# inference.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase-2 submission: uses InvestIQEnv directly (no HTTP round-trip) so the
# [START] / [STEP] / [END] blocks always reach stdout regardless of whether
# an external service is reachable.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json

# ── flush helper ──────────────────────────────────────────────────────────────
def emit(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()

# ── Groq / OpenAI client (optional — falls back gracefully) ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv not required; env vars may already be set

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")

try:
    from openai import OpenAI
    _api_key = GROQ_API_KEY or HF_TOKEN
    client = OpenAI(api_key=_api_key, base_url=API_BASE_URL) if _api_key else None
except Exception:
    client = None

# ── Import the environment directly ──────────────────────────────────────────
try:
    from environment.investiq_env import InvestIQEnv, PortfolioAction
    ENV_AVAILABLE = True
except Exception as _env_err:
    ENV_AVAILABLE = False
    emit(f"# WARNING: could not import InvestIQEnv: {_env_err}")

# ── Fallback stocks ───────────────────────────────────────────────────────────
FALLBACK_STOCKS = ["BHARTIARTL.NS", "NTPC.NS", "M&M.NS", "SUNPHARMA.NS"]


# ─────────────────────────────────────────────────────────────────────────────
# Agent logic
# ─────────────────────────────────────────────────────────────────────────────

def get_fallback_action(risk_score: int) -> dict:
    if risk_score < 40:
        return {"equity_pct": 25.0, "debt_pct": 65.0, "gold_pct": 10.0,
                "selected_stocks": FALLBACK_STOCKS}
    elif risk_score < 70:
        return {"equity_pct": 55.0, "debt_pct": 35.0, "gold_pct": 10.0,
                "selected_stocks": FALLBACK_STOCKS}
    else:
        return {"equity_pct": 70.0, "debt_pct": 20.0, "gold_pct": 10.0,
                "selected_stocks": FALLBACK_STOCKS}


def agent_decide(state, task_id: str) -> dict:
    """
    Ask the LLM for an allocation decision.
    Falls back to rule-based logic if the LLM is unavailable.
    Works with both EnvironmentState objects and plain dicts.
    """
    try:
        # Support both object and dict forms of state
        if isinstance(state, dict):
            user        = state["user"]
            market_data = state.get("market_data", [])
            risk_score  = user["risk_score"] if isinstance(user, dict) else user.risk_score
        else:
            user        = state.user
            market_data = state.market_data
            risk_score  = user.risk_score

        risk_level = ("conservative" if risk_score < 40 else
                      "balanced"     if risk_score < 70 else
                      "aggressive")

        # Normalise market_data items to dict
        def to_dict(m):
            return m if isinstance(m, dict) else m.dict()

        md_list    = [to_dict(m) for m in market_data]
        top_stocks = sorted(md_list, key=lambda x: x["sharpe"], reverse=True)[:8]

        stock_summary = "\n".join([
            f"  {s['ticker']}: sharpe={s['sharpe']:.2f}, "
            f"momentum={s['momentum']:.1f}%, sector={s['sector']}"
            for s in top_stocks
        ])

        # ── Try LLM ──────────────────────────────────────────────────────────
        if client is not None:
            invest_amount = (
                user["investment_amount"]
                if isinstance(user, dict)
                else user.investment_amount
            )
            prompt = f"""You are an Indian investment advisor.

User: risk_score={risk_score}/100 ({risk_level}), amount=₹{invest_amount:,.0f}

Top stocks by Sharpe:
{stock_summary}

Rules:
- equity_pct + debt_pct + gold_pct = exactly 100
- conservative (risk<40): equity<=30, debt>=50
- balanced (risk 40-70): equity 40-60, debt 20-40
- aggressive (risk>70): equity>=60, debt<=20
- Pick 4 stocks from the list above

Respond ONLY with this exact JSON (no markdown, no explanation):
{{"equity_pct": 60, "debt_pct": 30, "gold_pct": 10, "selected_stocks": ["T1.NS","T2.NS","T3.NS","T4.NS"]}}"""

            response = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 150,
                temperature = 0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            for tag in ["```json", "```"]:
                if tag in raw:
                    raw = raw.split(tag)[1].split("```")[0].strip()
                    break

            action = json.loads(raw)

            # Normalise percentages to 100
            e = float(action.get("equity_pct", 60))
            d = float(action.get("debt_pct",   30))
            g = float(action.get("gold_pct",   10))
            total = e + d + g
            if total > 0:
                e = round(e / total * 100, 1)
                d = round(d / total * 100, 1)
                g = round(100 - e - d, 1)

            # Validate stock tickers
            available = {m["ticker"] for m in md_list}
            stocks    = [s for s in action.get("selected_stocks", [])
                         if s in available]

            if len(stocks) < 4:
                for s in top_stocks:
                    if s["ticker"] not in stocks:
                        stocks.append(s["ticker"])
                    if len(stocks) == 4:
                        break

            return {"equity_pct": e, "debt_pct": d, "gold_pct": g,
                    "selected_stocks": stocks[:4]}

        # ── No LLM: rule-based from top stocks ───────────────────────────────
        stocks = [s["ticker"] for s in top_stocks[:4]]
        return {**get_fallback_action(risk_score), "selected_stocks": stocks}

    except Exception as ex:
        emit(f"# agent_decide error: {ex}")
        try:
            rs = (state["user"]["risk_score"]
                  if isinstance(state, dict)
                  else state.user.risk_score)
        except Exception:
            rs = 50
        return get_fallback_action(rs)


# ─────────────────────────────────────────────────────────────────────────────
# Task runner — direct in-process calls (no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def run_task_direct(task_id: str, max_steps: int) -> float:
    """Run one task using InvestIQEnv directly (no HTTP)."""
    emit(f"[START] task={task_id}")

    final_score = 0.0
    step_num    = 0

    try:
        env   = InvestIQEnv(task_id)
        state = env.reset()

        for step_num in range(1, max_steps + 1):
            try:
                action_dict = agent_decide(state, task_id)
            except Exception:
                rs = state.user.risk_score if hasattr(state, "user") else 50
                action_dict = get_fallback_action(rs)

            try:
                portfolio_action = PortfolioAction(**action_dict)
                result      = env.step(portfolio_action)
                reward      = float(result.reward)
                done        = result.done
                final_score = float(result.state.score_so_far)
                state       = result.state
            except Exception as ex:
                emit(f"# step error: {ex}")
                reward      = 0.0
                done        = True
                final_score = 0.0

            emit(f"[STEP] step={step_num} reward={reward}")

            if done:
                break

    except Exception as ex:
        emit(f"# task error: {ex}")
        if step_num == 0:
            emit(f"[STEP] step=1 reward=0.0")
            step_num = 1
        final_score = 0.0

    emit(f"[END] task={task_id} score={final_score} steps={step_num}")
    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Fallback runner — HTTP (used only when env import failed)
# ─────────────────────────────────────────────────────────────────────────────

def run_task_http(task_id: str, max_steps: int, env_url: str) -> float:
    """Fallback: hit a running HTTP server."""
    import requests as _req

    emit(f"[START] task={task_id}")

    final_score = 0.0
    step_num    = 0

    try:
        res   = _req.post(f"{env_url}/reset", params={"task_id": task_id}, timeout=60)
        res.raise_for_status()
        state = res.json()

        for step_num in range(1, max_steps + 1):
            try:
                action = agent_decide(state, task_id)
            except Exception:
                action = get_fallback_action(state.get("user", {}).get("risk_score", 50))

            try:
                r      = _req.post(
                    f"{env_url}/step",
                    params  = {"task_id": task_id},
                    json    = action,
                    headers = {"Content-Type": "application/json"},
                    timeout = 60,
                )
                r.raise_for_status()
                result      = r.json()
                reward      = float(result.get("reward", 0.0))
                done        = result.get("done", True)
                final_score = float(result["state"].get("score_so_far", reward))
                state       = result["state"]
            except Exception as ex:
                emit(f"# step error: {ex}")
                reward      = 0.0
                done        = True
                final_score = 0.0

            emit(f"[STEP] step={step_num} reward={reward}")

            if done:
                break

    except Exception as ex:
        emit(f"# task error: {ex}")
        if step_num == 0:
            emit(f"[STEP] step=1 reward=0.0")
            step_num = 1
        final_score = 0.0

    emit(f"[END] task={task_id} score={final_score} steps={step_num}")
    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    emit("# InvestIQ OpenEnv — Phase-2 Inference")
    emit(f"# Model: {MODEL_NAME}")
    emit(f"# ENV_AVAILABLE: {ENV_AVAILABLE}")

    tasks = [
        ("task1_allocation",      1),
        ("task2_stock_selection", 2),
        ("task3_full_portfolio",  3),
    ]

    scores = {}

    env_url = os.getenv("ENV_URL", "")

    for task_id, max_steps in tasks:
        if ENV_AVAILABLE:
            score = run_task_direct(task_id, max_steps)
        else:
            # Last resort: try HTTP if ENV_URL is set
            if env_url:
                score = run_task_http(task_id, max_steps, env_url)
            else:
                # Absolute last resort: emit minimal valid blocks with 0 score
                emit(f"[START] task={task_id}")
                emit(f"[STEP] step=1 reward=0.0")
                emit(f"[END] task={task_id} score=0.0 steps=1")
                score = 0.0

        scores[task_id] = score

    emit("\n# === FINAL SCORES ===")
    for tid, sc in scores.items():
        emit(f"# {tid}: {sc}")

    avg = sum(scores.values()) / len(scores)
    emit(f"# Average: {avg:.4f}")


if __name__ == "__main__":
    main()