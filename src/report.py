import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from datetime import datetime

from src.metrics import TRADING_DAYS, rolling_sharpe


def _normalized(prices_df: pd.DataFrame) -> pd.DataFrame:
    return prices_df / prices_df.iloc[0]

def _drawdown(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1

# Render mini risk report
def _build_text_panel(
    latest_table: pd.DataFrame, 
    rf_annual: float, 
    window: int, 
    top_n: int, 
    benchmark: str | None = None
) -> str:
    show = latest_table.copy()
    show = show.head(min(top_n, len(show)))

    var_col = next((c for c in latest_table.columns if c.startswith("VaR")), None)
    cvar_col = next((c for c in latest_table.columns if c.startswith("CVaR")), None)
    exc_col = next((c for c in latest_table.columns if c.startswith("VaRExceed")), None)
    sharpe_col = next((c for c in latest_table.columns if c.startswith("RollingSharpe")), None)
    beta_col = next((c for c in latest_table.columns if c.startswith("Beta")), None)
    corr_col = next((c for c in latest_table.columns if c.startswith("Corr")), None)

    var_level_str = "NA"
    if var_col:
        var_level_str = "".join(ch for ch in var_col if ch.isdigit()) or "NA"
    
    # Title
    lines = []
    lines.append("LATEST METRICS (Highest Risk First)")
    info_line = (f"Window={window}D | VaR level={var_level_str} | rf={rf_annual:.2%}")
    if benchmark:
        info_line += (f" | Benchmark={benchmark}")
    lines.append(info_line)
    lines.append("-" * 130)

    # Column Header
    header = (
        f"{'Rank':>4}  {'Ticker':<8}  {'Reg':<4}  "
        f"{'LastPx':>8}  {'AnnVol':>7}  {'MaxDD':>7}  "
        f"{'VaR':>7}  {'CVaR':>7}  {'Excd':>6}  {'Sh':>7}"
    )
    if beta_col:
        header += f"  {'Beta':>6}"
    if corr_col:
        header += f"  {'Corr':>6}"
    header += "  Alerts"
    lines.append(header)
    lines.append("-" * 130)
    
    # Rows
    for i, (_, r) in enumerate(show.iterrows(), start=1):
        last_px = float(r.get("LastPrice", np.nan))
        ann_vol = float(r.get("AnnVol", np.nan))
        maxdd = float(r.get("MaxDD", np.nan))
        regime = str(r.get("VolRegime", "NA"))

        var_v = float(r[var_col]) if var_col else np.nan
        cvar_v = float(r[cvar_col]) if cvar_col else np.nan
        exc_v = float(r[exc_col]) if exc_col else np.nan
        sh_v = float(r[sharpe_col]) if sharpe_col else np.nan

        beta_v = float(r[beta_col]) if beta_col else np.nan
        corr_v = float(r[corr_col]) if corr_col else np.nan

        alerts = str(r.get("Alerts", "")).strip() or "-"

        row = (
            f"{i:>4}  {r['Ticker']:<8}  {regime:<4}  "
            f"{last_px:>8.2f}  {ann_vol:>7.1%}  {maxdd:>7.1%}  "
            f"{var_v:>7.1%}  {cvar_v:>7.1%}  {exc_v:>6.1%}  {sh_v:>7.2f}"
        )
        if beta_col:
            row += f"  {beta_v:>6.2f}"
        if corr_col:
            row += f"  {corr_v:>6.2f}"
        row += f"  {alerts}"
        lines.append(row)

    return "\n".join(lines)

def _plot_corr_heatmap(ax: plt.Axes, rets_df: pd.DataFrame, lookback: int) -> None:
    if rets_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Correlation Heatmap Unavailable [No returns]", ha="center", va="center")
        return
    
    use = rets_df.tail(lookback).dropna(how="any")
    if use.shape[0] < 10 or use.shape[1] < 2:
        ax.axis("off")
        ax.text(0.5, 0.5, "Correlation HeatMap Unavailable [Too little data]", ha="center", va="center")
        return
    
    corr = use.corr().fillna(0.0).clip(-1.0, 1.0)

    dist = (1.0 - corr).clip(0.0, 2.0)
    np.fill_diagonal(dist.values, 0.0)
    Z = linkage(squareform(dist.values, checks=False), method="average")
    order = leaves_list(Z)

    corr_ord = corr.iloc[order, order]
    labels = list(corr_ord.columns)

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(corr.values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(f"Return Correlation Heatmap (last ~{min(lookback, use.shape[0])} days)", pad=10)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=90)

def make_risk_radar_figure(
    prices_df: pd.DataFrame,
    latest_table: pd.DataFrame,
    rf_annual: float = 0.0,
    window: int = 63,
    top_n: int = 8,
    corr_lookback: int = 252,
    benchmark: str | None = None,
    exclude_benchmark_from_plots: bool = True
):
    rets = prices_df.pct_change().dropna()

    prices_assets = prices_df
    rets_assets = rets
    if benchmark and benchmark in prices_df.columns and exclude_benchmark_from_plots:
        prices_assets = prices_df.drop(columns=[benchmark])
        rets_assets = rets.drop(columns=[benchmark])

    norm = _normalized(prices_assets)
    dd_df = prices_assets.apply(_drawdown)

    roll_vol = rets_assets.rolling(window).std() * np.sqrt(TRADING_DAYS)
    roll_sh = rolling_sharpe(rets_assets, rf_annual, window)

    fig = plt.figure(figsize=(24, 14), dpi=400)

    A = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.35, wspace=0.22)

    ax_price = fig.add_subplot(A[0, 0])
    ax_dd = fig.add_subplot(A[0, 1])
    ax_vol = fig.add_subplot(A[1, 0])
    ax_sh = fig.add_subplot(A[1, 1])
    ax_heat = fig.add_subplot(A[2, 0])
    ax_text = fig.add_subplot(A[2, 1])
    ax_text.axis("off")

    # Show dates without clutter
    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in [ax_price, ax_dd, ax_vol, ax_sh]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    norm.plot(ax=ax_price, linewidth=1.2, legend=True)
    ax_price.set_title("Normalized Price (Start=1.0)", pad=10)
    ax_price.grid(True)

    dd_df.plot(ax=ax_dd, linewidth=1.2, legend=True)
    ax_dd.set_title("Drawdown", pad=10)
    ax_dd.grid(True)

    roll_vol.plot(ax=ax_vol, linewidth=1.2, legend=True)
    ax_vol.set_title(f"Rolling Annualized Vol ({window}D)", pad=15)
    ax_vol.grid(True)

    roll_sh.plot(ax=ax_sh, linewidth=1.2, legend=True)
    ax_sh.set_title(f"Rolling Sharpe ({window}D), rf={rf_annual:.2%}", pad=15)
    ax_sh.grid(True)

    _plot_corr_heatmap(ax_heat, rets_assets, lookback=corr_lookback)

    handles, labels = ax_price.get_legend_handles_labels()
    for ax in (ax_price, ax_dd, ax_vol, ax_sh):
        ax.set_xlabel("")
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=min(8, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 0.95)
    )

    text_block = _build_text_panel(latest_table, rf_annual, window, top_n, benchmark)
    ax_text.text(
        0.01, 0.98,
        text_block,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace"
    )

    fig.suptitle("Equity Risk Radar (Regimes + Alerts)", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.75])

    return fig

def save_outputs(fig, latest_table: pd.DataFrame):
    os.makedirs("reports", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = f"reports/latest_metrics_{stamp}.csv"
    png_path = f"reports/risk_radar_{stamp}.png"

    latest_table.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=200)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {png_path}")


