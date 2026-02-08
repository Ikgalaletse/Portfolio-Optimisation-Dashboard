# app.py
# Mean-Variance + Capital Allocation Line (CAL) Streamlit Dashboard (Dark Mode, uncluttered)

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.optimize import minimize

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Mean–Variance + Capital Allocation (CAL)",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Dark mode CSS (forces dark background + readable components)
# ---------------------------
st.markdown(
    """
<style>
/* ===============================
   GLOBAL APP BACKGROUND
================================ */
.stApp {
    background-color: #0e1117;
    color: #ffffff !important;
}

/* ===============================
   TOP HEADER BAR (Streamlit chrome)
================================ */
header[data-testid="stHeader"] {
    background-color: #0e1117 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}

/* ===============================
   TEXT & HEADINGS
================================ */
h1, h2, h3, h4, h5, h6,
p, span, label, div {
    color: #ffffff !important;
}

/* Secondary/helper text */
.small-note {
    font-size: 0.9rem;
    opacity: 0.9;
    color: #e5e7eb !important;
}

/* ===============================
   INPUT BOXES (MAKE THEM WHITE)
================================ */
input, textarea, select {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Streamlit number inputs / text inputs */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Sliders */
div[data-baseweb="slider"] {
    color: #ffffff !important;
}

/* ===============================
   BUTTONS
================================ */
.stButton > button {
    background-color: #ef4444;
    color: #ffffff !important;
    border-radius: 12px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #dc2626;
}

/* ===============================
   DATAFRAMES / TABLES
================================ */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background-color: #0b1220 !important;
    color: #ffffff !important;
    border-radius: 12px;
}

/* ===============================
   TABS
================================ */
button[data-baseweb="tab"] {
    color: #ffffff !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #ef4444 !important;
}

/* ===============================
   PLOTS (MATPLOTLIB CONTAINERS)
================================ */
.stPyplotContainer {
    background-color: transparent !important;
}

/* ===============================
   REMOVE EXTRA TOP PADDING
================================ */
.block-container {
    padding-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
TRADING_DAYS = 252


def _clean_tickers(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in raw.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False)
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in data.columns:
                closes[t] = data[(t, "Close")]
            elif (t, "Adj Close") in data.columns:
                closes[t] = data[(t, "Adj Close")]
        prices = pd.DataFrame(closes)
    else:
        col = "Close" if "Close" in data.columns else ("Adj Close" if "Adj Close" in data.columns else None)
        if col is None:
            return pd.DataFrame()
        prices = data[[col]].rename(columns={col: tickers[0]})

    prices = prices.sort_index().dropna(how="all")
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualize_mean_cov(daily_returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu_daily = daily_returns.mean()
    cov_daily = daily_returns.cov()
    mu_ann = mu_daily * TRADING_DAYS
    cov_ann = cov_daily * TRADING_DAYS
    return mu_ann, cov_ann


def tangency_weights_long_only(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.Series:
    """
    Long-only tangency portfolio:
    - No short selling (w >= 0)
    - Fully invested (sum w = 1)
    """

    n = len(mu)
    excess = mu.values - rf
    cov_mat = cov.values

    # Objective: maximize Sharpe → minimize negative Sharpe numerator
    def objective(w):
        return -np.dot(w, excess)

    # Constraint: fully invested
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    # Bounds: no short selling
    bounds = tuple((0.0, 1.0) for _ in range(n))

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "disp": False},
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    return pd.Series(result.x, index=mu.index)


def portfolio_mu_vol(mu: pd.Series, cov: pd.DataFrame, w: pd.Series) -> Tuple[float, float]:
    wv = w.values.reshape(-1, 1)
    mu_p = float(np.dot(w.values, mu.values))
    var_p = float((wv.T @ cov.values @ wv)[0, 0])
    vol_p = float(np.sqrt(max(var_p, 0.0)))
    return mu_p, vol_p


def cal_allocation(
    risky_mu: float,
    risky_vol: float,
    rf: float,
    target_vol: Optional[float] = None,
    target_return: Optional[float] = None,
) -> float:
    if risky_vol <= 0:
        return 0.0
    if target_vol is not None:
        return float(target_vol / risky_vol)
    if target_return is not None:
        denom = (risky_mu - rf)
        if abs(denom) < 1e-12:
            return 0.0
        return float((target_return - rf) / denom)
    return 1.0


def quarterly_periods(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.to_period("Q").astype(str).values, index=index)


def backtest_quarterly_rebalanced(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    returns = returns.dropna(how="any")
    weights = weights.reindex(returns.columns)

    period = quarterly_periods(returns.index)

    port_rets = []
    last_period = None
    w = weights.values.copy()

    for dt, rvec, p in zip(returns.index, returns.values, period.values):
        if last_period is None:
            last_period = p
        elif p != last_period:
            w = weights.values.copy()
            last_period = p
        port_rets.append(float(np.dot(w, rvec)))

    port_rets = pd.Series(port_rets, index=returns.index, name="portfolio_return")
    wealth = (1.0 + port_rets).cumprod()
    wealth.name = "wealth"
    return wealth


def compute_performance(wealth: pd.Series) -> dict:
    if wealth.empty:
        return {}

    daily = wealth.pct_change().dropna()
    if len(daily) == 0:
        return {}

    cagr = (wealth.iloc[-1] ** (TRADING_DAYS / len(daily))) - 1
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    max_dd = ((wealth / wealth.cummax()) - 1).min()

    return {"CAGR": cagr, "AnnVol": vol, "MaxDD": max_dd, "Final": float(wealth.iloc[-1])}


def plot_growth(curves: List[Tuple[str, pd.Series]]) -> plt.Figure:
    # Dark-friendly Matplotlib
    plt.rcParams["axes.facecolor"] = "#0e1117"
    plt.rcParams["figure.facecolor"] = "#0e1117"
    plt.rcParams["savefig.facecolor"] = "#0e1117"
    plt.rcParams["text.color"] = "#e6e6e6"
    plt.rcParams["axes.labelcolor"] = "#e6e6e6"
    plt.rcParams["xtick.color"] = "#cbd5e1"
    plt.rcParams["ytick.color"] = "#cbd5e1"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["grid.color"] = "#334155"

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, s in curves:
        ax.plot(s.index, s.values, label=label, linewidth=2.0)
    ax.set_title("Growth of $1 (Quarterly Rebalanced)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------
# Sidebar (uncluttered + author)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Portfolio Setup")

    risky_raw = st.text_area(
        "Risky assets (tickers, comma-separated)",
        value="VTI, TLT, GLD",
        height=70,
        help="Example: VTI, VEA, VWO, TLT",
    )
    benchmark = st.text_input("Benchmark ticker", value="SPY").strip().upper()

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input("End", value=pd.to_datetime("2025-01-31")).strftime("%Y-%m-%d")

    st.divider()

    rf_annual = st.number_input(
        "Risk-free rate (annual)",
        value=0.05,
        min_value=-0.05,
        max_value=0.30,
        step=0.005,
        format="%.4f",
        help="Example: 0.05 = 5% per year",
    )

    objective = st.radio(
        "CAL target",
        ["Target volatility", "Target return"],
        help="Choose how to size the risky allocation (y).",
    )

    if objective == "Target volatility":
        target_vol = st.slider("Target annual volatility", 0.02, 0.50, 0.12, 0.01)
        target_return = None
    else:
        target_return = st.slider("Target annual return", -0.10, 0.40, 0.12, 0.01)
        target_vol = None

    st.divider()
    run = st.button("Run backtest", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Ikgalaletse Keatlegile Neo Sebola**")
    st.markdown("🔗 LinkedIn: https://www.linkedin.com/in/neo-sebola-499b72313/")
    st.caption("Built with Streamlit • Mean–Variance + CAL")

# ---------------------------
# Main (minimal + not cluttered)
# ---------------------------
st.title("Mean–Variance Portfolio Optimisation + Capital Allocation Line (CAL)")
st.markdown(
    '<div class="small-note">Optimizes a tangency risky portfolio from historical returns, '
    'then blends it with a risk-free asset using the Capital Allocation Line. '
    'Backtests use <b>quarterly rebalancing</b> and compare to a benchmark.</div>',
    unsafe_allow_html=True,
)

risky_tickers = _clean_tickers(risky_raw)
if not risky_tickers:
    st.warning("Enter at least one risky asset ticker.")
    st.stop()
if not benchmark:
    st.warning("Enter a benchmark ticker.")
    st.stop()

if not run:
    st.info("Configure inputs in the sidebar, then click **Run backtest**.")
    st.stop()

all_tickers = sorted(set(risky_tickers + [benchmark]))

with st.spinner("Downloading prices..."):
    prices = download_prices(all_tickers, start, end)

if prices.empty:
    st.error("No price data returned. Check tickers and dates.")
    st.stop()

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"Dropped tickers with no data: {', '.join(missing)}")
    prices = prices.drop(columns=missing, errors="ignore")

if benchmark not in prices.columns:
    st.error("Benchmark data missing. Choose a different benchmark.")
    st.stop()

risky_cols = [t for t in risky_tickers if t in prices.columns]
if len(risky_cols) == 0:
    st.error("None of the risky tickers returned data.")
    st.stop()

prices_risky = prices[risky_cols].dropna()
prices_bench = prices[[benchmark]].reindex(prices_risky.index).dropna()

# Align
prices_risky = prices_risky.loc[prices_bench.index]
prices_bench = prices_bench.loc[prices_risky.index]

rets_risky = to_returns(prices_risky)
rets_bench = to_returns(prices_bench).rename(columns={benchmark: "benchmark"})

common = rets_risky.index.intersection(rets_bench.index)
rets_risky = rets_risky.loc[common]
rets_bench = rets_bench.loc[common]

mu_ann, cov_ann = annualize_mean_cov(rets_risky)
w_tan = tangency_weights_long_only(mu_ann, cov_ann, rf_annual)
tan_mu, tan_vol = portfolio_mu_vol(mu_ann, cov_ann, w_tan)

y = cal_allocation(tan_mu, tan_vol, rf_annual, target_vol=target_vol, target_return=target_return)
y_capped = float(np.clip(y, -1.0, 2.0))
rf_weight = 1.0 - y_capped

# Backtests
wealth_tan = backtest_quarterly_rebalanced(rets_risky, w_tan)

rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
tan_daily = wealth_tan.pct_change().dropna()
cal_daily = rf_daily + y_capped * (tan_daily - rf_daily)
wealth_cal = (1.0 + cal_daily).cumprod()
wealth_bench = (1.0 + rets_bench["benchmark"]).cumprod()

# Align wealth series
min_start = max(wealth_tan.index.min(), wealth_cal.index.min(), wealth_bench.index.min())
wealth_tan = wealth_tan.loc[min_start:]
wealth_cal = wealth_cal.loc[min_start:]
wealth_bench = wealth_bench.loc[min_start:]

# ---------------------------
# Compact KPI row (only what matters)
# ---------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tangency μ (ann.)", f"{tan_mu:.2%}")
k2.metric("Tangency σ (ann.)", f"{tan_vol:.2%}")
k3.metric("Risky allocation (y)", f"{y_capped:.1%}")
k4.metric("Risk-free allocation", f"{rf_weight:.1%}")

# ---------------------------
# Two tabs: Plot + Allocations (keeps UI clean)
# ---------------------------
tab_plot, tab_alloc = st.tabs(["📊 Backtest Plot", "📌 Allocations"])

with tab_plot:
    fig = plot_growth(
        [
            ("CAL Portfolio", wealth_cal),
            ("Tangency (Risky)", wealth_tan),
            (f"Benchmark ({benchmark})", wealth_bench),
        ]
    )
    st.pyplot(fig, use_container_width=True)

    # Minimal performance table (compact)
    perf = pd.DataFrame(
        {
            "Portfolio": ["CAL", "Tangency", f"Benchmark ({benchmark})"],
            "CAGR": [
                compute_performance(wealth_cal)["CAGR"],
                compute_performance(wealth_tan)["CAGR"],
                compute_performance(wealth_bench)["CAGR"],
            ],
            "Ann. Vol": [
                compute_performance(wealth_cal)["AnnVol"],
                compute_performance(wealth_tan)["AnnVol"],
                compute_performance(wealth_bench)["AnnVol"],
            ],
            "Max DD": [
                compute_performance(wealth_cal)["MaxDD"],
                compute_performance(wealth_tan)["MaxDD"],
                compute_performance(wealth_bench)["MaxDD"],
            ],
        }
    )
    st.dataframe(
        perf.style.format({"CAGR": "{:.2%}", "Ann. Vol": "{:.2%}", "Max DD": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_alloc:
    st.markdown("### Capital Allocation Line (CAL)")
    st.write("**Allocation between risk-free asset and tangency risky portfolio**")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": ["Risk-free asset", "Tangency risky portfolio"],
                "Weight": [rf_weight, y_capped],
            }
        ).style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Weights inside the tangency risky portfolio**")
    w_df = pd.DataFrame({"Asset": w_tan.index, "Weight": w_tan.values}).sort_values("Weight", ascending=False)
    st.dataframe(
        w_df.style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Notes", expanded=False):
    st.markdown(
        """
- Tangency weights are **unconstrained** (weights may be negative).
- Backtest uses **quarterly rebalancing**.
- Risk-free rate is assumed **constant**.
- If **y > 100%**, it implies leverage; if **y < 0%**, it implies shorting the risky portfolio.
"""
    )