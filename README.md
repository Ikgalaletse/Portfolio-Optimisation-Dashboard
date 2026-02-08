# 🌙 Mean–Variance Portfolio Optimization with Capital Allocation Line (CAL)

This repository contains an **interactive Streamlit dashboard** that implements **mean–variance portfolio optimization** combined with the **Capital Allocation Line (CAL)** framework.

The dashboard is designed to help users **understand portfolio construction**, **asset allocation**, and **risk–return trade-offs** in a clear, visual, and practical way using historical data.

---

## 📌 What This Dashboard Does

The dashboard performs the following steps:

1. **Constructs a risky (tangency) portfolio**
   - Uses historical returns of user-selected assets
   - Applies **mean–variance optimization**
   - Enforces **long-only constraints** (no short selling)
   - Fully invested portfolio (weights sum to 100%)

2. **Blends the risky portfolio with a risk-free asset**
   - Uses the **Capital Allocation Line (CAL)**
   - Determines how much to allocate to:
     - Risk-free asset
     - Risky (tangency) portfolio
   - Allocation can be set using:
     - Target volatility **or**
     - Target return

3. **Backtests the strategy**
   - Quarterly rebalancing
   - Compares performance against a benchmark (e.g. SPY)
   - Displays growth of \$1 over time

4. **Visualizes results clearly**
   - Portfolio growth plot
   - Allocation breakdown (risk-free vs risky)
   - Weights of assets inside the risky portfolio
   - Key performance metrics (CAGR, volatility, drawdown)

---

## 🧠 Core Concepts Used

- **Mean–Variance Optimization (Markowitz)**
- **Tangency Portfolio**
- **Capital Allocation Line (CAL)**
- **Quarterly Rebalancing**
- **Long-Only Constraints (No Short Selling)**
- **Risk-Free Asset Allocation**

---

## ⚙️ Dashboard Features

### Inputs (Sidebar)
- List of risky assets (tickers)
- Benchmark ticker
- Start and end date
- Risk-free rate (annual)
- CAL target:
  - Target volatility **or**
  - Target return

### Outputs
- Growth of \$1 plot (CAL vs Tangency vs Benchmark)
- Allocation to:
  - Risk-free asset
  - Risky portfolio
- Asset weights inside the risky portfolio
- Performance metrics:
  - CAGR
  - Annualized volatility
  - Maximum drawdown

---

## 🚫 Portfolio Constraints

This dashboard enforces **realistic investment constraints**:

- ❌ No short selling
- ❌ No leverage inside the risky portfolio
- ✅ All asset weights ≥ 0
- ✅ Weights sum to 100%

---

## 🧮 Data & Methodology

- Price data sourced via **Yahoo Finance**
- Returns calculated from adjusted prices
- Mean and covariance are **annualized**
- Optimization solved using **constrained numerical optimization**
- Risk-free rate assumed constant
- Rebalancing performed **quarterly**

---

## ▶️ How to Run the Dashboard Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name