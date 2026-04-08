---
title: InvestIQ OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---





# InvestIQ OpenEnv 📈

InvestIQ is an AI-powered portfolio management environment built using OpenEnv principles.

It simulates real-world investment decisions where an agent allocates capital across equity, debt, and gold based on user risk profiles.

---

## 🚀 Features

- 3 Tasks:
  - task1_allocation → Asset allocation
  - task2_stock_selection → Stock picking
  - task3_full_portfolio → Full portfolio optimization

- Real NSE stock data (10+ years simulated)
- Feature engineering (Sharpe, momentum, volatility)
- Reward system based on:
  - Risk alignment (40%)
  - Stock quality (30%)
  - Diversification (20%)
  - Constraints (10%)

---

## 🧠 Architecture

- FastAPI backend
- OpenEnv-compatible API:
  - `/reset`
  - `/step`
  - `/state`
- Hugging Face Space deployment
- Lazy loading to prevent startup timeouts

---

## 🌐 Live Demo

👉 https://shivdahake5-investiq-env.hf.space

Swagger UI:
👉 https://shivdahake5-investiq-env.hf.space/docs

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload