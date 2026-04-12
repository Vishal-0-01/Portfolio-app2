# Flexi-Cap Portfolio Optimizer

A valuation-aware portfolio construction engine combining Markowitz optimization with macro valuation signals (Nifty PE/PB).

Built by Vishal Singh.

---

## What this does

- Optimizes portfolio allocation across mutual funds
- Dynamically adjusts equity exposure based on valuation (PE/PB)
- Enforces portfolio-level risk constraints (volatility cap)
- Generates efficient frontier
- Runs backtests including walk-forward simulation

---

## Core Model Logic

### 1. Portfolio Optimization
- Mean-variance optimization (Markowitz)
- Objective: maximize return under volatility constraint
- Portfolio volatility cap: 10%

### 2. Valuation Overlay
- Uses Nifty 50 PE and PB
- Converts to z-scores
- Adjusts equity allocation:

| Valuation | Equity Allocation |
|----------|------------------|
| Cheap    | ~90%             |
| Fair     | ~70%             |
| Expensive| ~50%             |

### 3. Risk Modelling
- Covariance-based risk estimation
- Diversification across funds
- Portfolio-level constraint instead of per-asset restriction

### 4. Backtesting
- Allocation backtest (historical PE/PB driven)
- Performance simulation
- Walk-forward (no lookahead bias)

---

## Tech Stack

- Backend: Python (Flask)
- Optimization: NumPy / SciPy
- Data: yfinance / processed NAV data
- Frontend: HTML + JS + Chart.js

---

## API Endpoints

- `/api/optimize?pe=22&pb=3.5`
- `/api/funds`
- `/api/frontier`
- `/api/backtest`
- `/api/backtest-performance`
- `/api/backtest-walkforward`
- `/health`

---

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
