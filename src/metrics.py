import numpy as np
import pandas as pd

TRADING_DAYS = 252

def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def annualized_vol(rets: pd.Series) -> float:
    return float(rets.std() * np.sqrt(TRADING_DAYS))

def rolling_sharpe(rets: pd.Series | pd.DataFrame, rf_annual: float, window: int) -> pd.Series | pd.DataFrame:
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    m = excess.rolling(window).mean()
    s = rets.rolling(window).std().replace(0, np.nan)
    return (m / s) * np.sqrt(TRADING_DAYS)

def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    dd = prices / peak - 1
    return float(dd.min())

def hist_var(rets: pd.Series, level: float) -> float:
    q = 1 - level
    return float(rets.quantile(q))

def cvar_es(rets: pd.Series, level: float) -> float:
    q = 1 - level
    var = rets.quantile(q)
    tail = rets[rets <= var]
    if tail.empty:
        return float("nan")
    return float(tail.mean())

def var_exceedance_rate(rets: pd.Series, var_value: float) -> float:
    if rets.empty or np.isnan(var_value):
        return float("nan")
    return float((rets <= var_value).mean())

def vol_regime(roll_vol: pd.Series, q_low: float, q_high: float) -> str:
    v = roll_vol.dropna()

    if v.empty:
        return "NA"
    
    low = v.quantile(q_low)
    high = v.quantile(q_high)
    last = float(v.iloc[-1])

    if last <= low:
        return "LOW"
    if last >= high:
        return "HIGH"
    return "MID"

def _align_pair(asset_rets: pd.Series, bench_rets: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([asset_rets, bench_rets], axis=1, join="inner").dropna()
    if df.shape[0] < window:
        return (pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float))
    return df.iloc[:, 0], df.iloc[:, 1]

def rolling_beta(asset_rets: pd.Series, bench_rets: pd.Series, window: int) -> pd.Series:
    a, b = _align_pair(asset_rets, bench_rets, window)
    if a.empty:
        return a
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var().replace(0, np.nan)
    return cov / var

def rolling_corr(asset_rets: pd.Series, bench_rets: pd.Series, window: int) -> pd.Series:
    a, b = _align_pair(asset_rets, bench_rets, window)
    if a.empty:
        return a
    return a.rolling(window).corr(b)

def build_latest_metrics_table(
    prices_df: pd.DataFrame,
    rf_annual: float = 0.0,
    var_level: float = 0.95,
    vol_q_low: float = 0.3,
    vol_q_high: float = 0.7,
    alert_maxdd: float = -0.25,
    alert_var: float = -0.03,
    window: int = 63,
    benchmark_prices: pd.Series | None = None
) -> pd.DataFrame:
    rows = []

    bench_rets = daily_returns(benchmark_prices) if benchmark_prices is not None else None

    for t in prices_df.columns:
        px = prices_df[t]
        rets = daily_returns(px)

        # Core Risk Metrics
        ann_vol = annualized_vol(rets)
        mdd = max_drawdown(px)
        
        # Tail Risk (VaR + CVaR)
        var_x = hist_var(rets, var_level)
        cvar_x = cvar_es(rets, var_level)

        # Sanity Check
        exceed = var_exceedance_rate(rets, var_x)

        # Regime based on Rolling Volatility (Quarterly)
        roll_vol_w = rets.rolling(window).std() * np.sqrt(TRADING_DAYS)
        regime = vol_regime(roll_vol_w, vol_q_low, vol_q_high)

        # Rolling Sharpe
        roll_sh = rolling_sharpe(rets, rf_annual, window)
        last_roll_sh = float(roll_sh.dropna().iloc[-1]) if not roll_sh.dropna().empty else np.nan

        # Benchmark-Relative Metrics
        beta_last = np.nan
        corr_last = np.nan
        if bench_rets is not None:
            b = rolling_beta(rets, bench_rets, window).dropna()
            c = rolling_corr(rets, bench_rets, window).dropna()
            beta_last = float(b.iloc[-1]) if not b.empty else np.nan
            corr_last = float(c.iloc[-1]) if not c.empty else np.nan

        # Alerts (Policy Thresholds)
        alert_flags = []
        if mdd <= alert_maxdd:
            alert_flags.append(f"DD <= {int(abs(alert_maxdd) * 100)}%")
        if var_x <= alert_var:
            alert_flags.append(f"VaR <= {abs(alert_var):.1%}")
        if regime == "HIGH":
            alert_flags.append("HIGH_VOL")
        
        row = {
            "Ticker": t,
            "LastPrice": float(px.iloc[-1]),
            "AnnVol": ann_vol,
            "MaxDD": mdd,
            f"VaR{int(var_level * 100)}": var_x,
            f"CVaR{int(var_level * 100)}": cvar_x,
            f"VaRExceed{int(var_level * 100)}": exceed,
            "VolRegime": regime,
            f"RollingSharpe{window}": last_roll_sh,
            "Alerts": " | ".join(alert_flags) if alert_flags else "",
        }

        if bench_rets is not None:
            row[f"Beta{window}"] = beta_last
            row[f"Corr{window}"] = corr_last
        
        rows.append(row)
    
    out = pd.DataFrame(rows)

    regime_rank = {"HIGH": 0, "MID": 1, "LOW": 2, "NA": 3}
    out["RegimeRank"] = out["VolRegime"].map(regime_rank).fillna(99).astype(int)
    out["HasAlert"] = out["Alerts"].astype(str).str.len().gt(0)

    # Sort Priority:
    # 1) HasAlert True first
    # 2) HIGH/MID/LOW/NA by RegimeRank
    # 3) Higher vol first
    out = out.sort_values(
        by=["HasAlert", "RegimeRank", "AnnVol"], 
        ascending=[False, True, False]
    ).drop(columns=["HasAlert", "RegimeRank"])

    return out
