# Equity Risk Radar!


A lightweight multi-asset risk dashboard that turns a list of tickers into a single "risk radar" report. Core analytics include rolling annualized volatility, rolling Sharpe ratio (using a daily risk-free rate derived from the annual rate), historical VaR/CVaR (Expected Shortfall) from empirical return quantiles, and drawdown / max drawdown based on peak-to-trough declines.


For market context, the tool optionally accepts a benchmark and estimates rolling CAPM beta (rolling covariance with benchmark / rolling benchmark variance) and rolling correlation, using synchronized return windows. It also computes a return correlation matrix over a configurable lookback period and renders it as a heatmap to highlight diversification and clustering effects among assets. Results are presented as a compact “Risk Radar” report: normalized price paths, drawdown trajectories, rolling vol and Sharpe, correlation heatmap, plus a ranked text panel summarizing the latest metrics and rule-based alerts. Outputs can be exported as PNG + CSV for sharing and downstream analysis.


Built with Python: Streamlit for the UI, yfinance for market data retrieval, pandas/numpy for time-series processing, SciPy for statistical utilities, and Matplotlib for report visualization. 


---

## Key Components and Metrics

**Annualized Volatility**&emsp; A measure of price fluctuations per year
<p align="center"><img src="assets/ann_vol.png" width="75%" height="75%"></img></p>

**Max Drawdown**&emsp; Worst peak-to-trough loss over a period
<p align="center"><img src="assets/max_dd.png" width="75%" height="75%"></img></p>

**Rolling Sharpe Ratio**&emsp; Risk-adjusted performance over the last W days
<p align="center"><img src="assets/roll_sharpe.png" width="75%" height="75%"></img></p>

**Historical Value-at-Risk (VaR)**&emsp; Worst-day loss threshold under historical behavior
<p align="center"><img src="assets/var.png" width="75%" height="75%"></img></p>

**Conditional Value-at-Risk (CVaR) / Expected Shortfall (ES)**&emsp; Average loss given you are in the worst tail beyond VaR
<p align="center"><img src="assets/cvar.png" width="75%" height="75%"></img></p>

**VaR Exceedance Rate**&emsp; How often returns breach VaR
<p align="center"><img src="assets/exc.png" width="75%" height="75%"></img></p>

**Return Correlation Matrix**&emsp; How strongly assets move together (gives insights on diversification / clustering risk)
<p align="center"><img src="assets/corr_matrix.png" width="75%" height="75%"></img></p>

**Rolling Beta vs Benchmark (CAPM)**&emsp; Tendency for assets to move when benchmark moves (measures market sensitivity)
<p align="center"><img src="assets/roll_beta.png" width="75%" height="75%"></img></p>

**Rolling Correlation vs Benchmark**&emsp; Strength of co-movement with benchmark (direction and tightness of linkage)
<p align="center"><img src="assets/roll_corr.png" width="75%" height="75%"></img></p>


---

## Background

### What exactly does this pipeline do?

1. Fetches historical price data (Adjusted Close/Close) via yfinance
2. Computes core risk metrics that are commonly used in buyside / risk teams
3. Classifies volatility regimes (LOW/MID/HIGH) using rolling volatility quantiles
4. Generates simple alerts when risk thresholds are breached
5. Accepts optional benchmark, computes each asset's beta and correlation vs benchmark and adds a return correlation heatmap
6. Exports two deliverables including:
    - A visual report showing time-series risk behavior
    - A metrics table summarizing the latest risk snapshot per ticker

### What are some of the default parameters?

- Trading Days = 252
- Rolling Window = 63
- Correlation Lookback = 252
- VaR Level = 0.95
- Risk-Free Rate (Annual) = 0.03

### What is a volatility regime classification?

Markets exhibit volatility clustering, as such risk teams and systematic strategies often explicitly model regimes like:
- risk-on vs risk-off
- low vol vs high vol
- normal vs stressed

In our case we are using quantile thresholds. After computing rolling annualized volatility σₜ(W), regimes are assigned using cross-sectional quantiles over time of σₜ(W) for each asset (e.g., LOW if σₜ(W) is below its own historical 30th percentile). This converts a continuous time series into an interpretable state label.

### How might this tool help?

The "radar" format makes it easy to answer the kinds of questions one might ask in portfolio monitoring or pre-trade checks:
- How risky is each asset today vs its own history?
- What's the worst recent downside profile?
- Am I being paid for the risk I'm taking?
- How much is this driven by the market/benchmark?
- Is my basket actually diversified?


---

## Try It Yourself
1) Install
```
python -m venv .venv
pip install -r requirements.txt
```

2) Run(CLI)
```
python run.py
```

This saves:
- reports/risk_radar_<timestamp>.png
- reports/latest_metrics_<timestamp>.csv

3) Run (Streamlit)
```
streamlit run app.py
```

Fully customizable parameters are on the sidebar.


---

## Notes

- This is a project done out of personal interest. It is not legal investment advice.
- Metrics are sensitive to data quality (missing days, corporate actions, liquidity, etc.).
- Historical VaR/CVaR assume the empirical distribution is informative for future risk.
- Results depend on lookback/window choices and are not robust to regime shifts.

