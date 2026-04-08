---
title: InvestIQ OpenEnv
emoji: 📈
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# InvestIQ Portfolio Management Environment

A real-world OpenEnv environment for Indian stock market
portfolio optimization. An AI agent learns to allocate
capital across equity, debt, and gold based on a user's
risk profile — evaluated against 10yr NSE market data.

## Real-world problem
97% of Indian investors have no access to personalized
financial advice. This environment trains agents to fill
that gap using real NSE data.

## Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| task1_allocation | Easy | 1 | Choose equity/debt/gold split |
| task2_stock_selection | Medium | 2 | Pick best NSE stocks |
| task3_full_portfolio | Hard | 3 | Full portfolio optimization |

## Observation Space
- risk_score: int (0-100)
- investment_amount: float
- income: str
- risk_appetite: str
- time_horizon: str
- goal: str
- market_data: list of stock features (sharpe, momentum, volatility)

## Action Space
- equity_pct: float (0-100)
- debt_pct: float (0-100)
- gold_pct: float (0-100)
- selected_stocks: list[str] (NSE tickers)

## Reward
Scores 0.0-1.0 based on:
- Allocation match to risk profile (40%)
- Stock quality by Sharpe ratio (30%)
- Sector diversification (20%)
- Constraint satisfaction (10%)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start fresh episode |
| /step | POST | Take an action |
| /state | GET | Get current state |
| /tasks | GET | List all tasks |

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run inference
```bash
python inference.py
```

## Environment Variables
- API_BASE_URL: LLM API endpoint
- MODEL_NAME: Model identifier
- HF_TOKEN: Hugging Face token
- GROQ_API_KEY: Groq API key
- ENV_URL: Environment URL (default: http://localhost:8000)