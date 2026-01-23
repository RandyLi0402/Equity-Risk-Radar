import io
import streamlit as st

from datetime import datetime
from src.fetch import get_price_panel
from src.metrics import build_latest_metrics_table
from src.report import make_risk_radar_figure

st.set_page_config(page_title="Equity Risk Radar", layout="wide")
st.title("Equity Risk Radar (Regimes + Alerts)")

def _parse_tickers(s: str) -> list[str]:
    parts = []
    for line in s.replace(",", "\n").splitlines():
        t = line.strip()
        if t:
            parts.append(t)

    
    out, seen = [], set()
    for t in parts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# Fetch Price Panel
@st.cache_data(ttl=3600)
def cached_price_panel(all_tickers: tuple[str, ...], start: str, end: str | None):
    return get_price_panel(list(all_tickers), start, end=end)

def _init_state() -> None:
    st.session_state.setdefault("ran", False)
    st.session_state.setdefault("png_bytes", None)
    st.session_state.setdefault("csv_bytes", None)
    st.session_state.setdefault("latest_table", None)
    st.session_state.setdefault("run_meta", None)

_init_state()

# Side Bar
st.sidebar.header("Parameters")

with st.sidebar.form("radar_form"):
    tickers_text = st.text_area(
        "Tickers (Separated by Comma / Newline)",
        value="0700.HK,9988.HK,9618.HK,3690.HK,1810.HK,2020.HK",
        height=120,
    )

    benchmark = st.text_input("Benchmark (Optional)", value="2800.HK").strip()
    benchmark = benchmark or None

    start = st.text_input("Start Date (YYYY-MM-DD)", value="2025-01-01")
    end_raw = st.text_input("End Date (Optional YYYY-MM-DD)", value="").strip()
    end = end_raw or None

    rf_annual = st.number_input("Risk-Free Rate (Annual)", value=0.03, step=0.005, format="%.3f")
    window = st.selectbox("Rolling Window (Days)", options=[21, 63, 126, 252], index=1)
    corr_lookback = st.selectbox("Correlation Lookback (Days)", options=[63, 126, 252, 504], index=2)
    var_level = st.selectbox("VaR Level", options=[0.90, 0.95, 0.99], index=1)

    vol_q_low = st.slider("Vol Regime Low Quantile", 0.05, 0.45, 0.30, 0.05)
    vol_q_high = st.slider("Vol Regime High Quantile", 0.55, 0.95, 0.70, 0.05)

    alert_maxdd = st.slider("Alert: MaxDD Threshold", -0.60, -0.05, -0.25, 0.01)
    alert_var = st.slider("Alert: VaR Threshold (Daily)", -0.10, -0.005, -0.03, 0.001)

    top_n = st.slider("Rows in Mini Report", 2, 16, 8, 1)

    run_btn = st.form_submit_button("Run Radar")

# Run Pipeline
if run_btn:
    tickers = _parse_tickers(tickers_text)
    if len(tickers) < 2:
        st.error("Please provide at least 2 tickers.")
        st.stop()

    all_tickers = tuple(tickers + ([benchmark] if benchmark else []))

    with st.spinner("Fetching prices..."):
        panel = cached_price_panel(all_tickers, start, end)
        prices = panel[tickers]
        bench_px = panel[benchmark] if benchmark else None

    with st.spinner("Computing metrics..."):
        latest_table = build_latest_metrics_table(
            prices,
            rf_annual=rf_annual,
            var_level=var_level,
            vol_q_low=vol_q_low,
            vol_q_high=vol_q_high,
            alert_maxdd=alert_maxdd,
            alert_var=alert_var,
            window=window,
            benchmark_prices=bench_px,
        )

    with st.spinner("Building report figure..."):
        fig = make_risk_radar_figure(
            prices,
            latest_table,
            rf_annual=rf_annual,
            window=window,
            top_n=top_n,
            corr_lookback=corr_lookback,
            benchmark=benchmark,
        )

    
    png_buf = io.BytesIO()
    # Preview may be scaled, but download is high-res
    fig.savefig(png_buf, format="png", dpi=300)  
    png_bytes = png_buf.getvalue()

    csv_bytes = latest_table.to_csv(index=False).encode("utf-8")

    
    st.session_state["ran"] = True
    st.session_state["png_bytes"] = png_bytes
    st.session_state["csv_bytes"] = csv_bytes
    st.session_state["latest_table"] = latest_table
    st.session_state["run_meta"] = {
        "benchmark": benchmark,
        "window": window,
        "corr_lookback": corr_lookback,
        "rf_annual": rf_annual,
        "var_level": var_level,
    }

    # Free Memory
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


if st.session_state.get("ran") and st.session_state.get("png_bytes") is not None:
    col1, col2 = st.columns([2, 1])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.image(st.session_state["png_bytes"], use_container_width=True)

        st.download_button(
            "Download Report PNG",
            data=st.session_state["png_bytes"],
            file_name="risk_radar_{stamp}.png",
            mime="image/png",
        )

    with col2:
        st.subheader("Latest Metrics")
        st.dataframe(st.session_state["latest_table"], use_container_width=True)

        st.download_button(
            "Download Metrics CSV",
            data=st.session_state["csv_bytes"],
            file_name="latest_metrics_{stamp}.csv",
            mime="text/csv",
        )

    meta = st.session_state.get("run_meta") or {}
    if meta.get("benchmark"):
        st.caption(f"Benchmark used for Beta/Corr columns: {meta['benchmark']}")
else:
    st.info("Set parameters in the sidebar, then click **Run Radar**.")
