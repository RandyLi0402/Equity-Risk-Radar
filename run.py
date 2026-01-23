from src.fetch import get_price_panel
from src.metrics import build_latest_metrics_table
from src.report import make_risk_radar_figure, save_outputs

DEFAULT_TICKERS = [
    "0700.HK",  # Tencent
    "9988.HK",  # Alibaba
    "9618.HK",  # JD.com     
    "3690.HK",  # Meituan
    "1810.HK",  # Xiaomi
    "2020.HK",  # ANTA Sports

]

DEFAULT_BENCHMARK = "2800.HK" # Hang Seng Tracker Fund

def main():
    # Configurations
    tickers = DEFAULT_TICKERS
    benchmark = DEFAULT_BENCHMARK
    
    start = "2025-01-01"
    end = None

    rf_annual = 0.03
    var_level = 0.95
    vol_q_low = 0.3
    vol_q_high = 0.7

    window = 63
    corr_lookback = 252

    # Alert Thresholds
    alert_maxdd = -0.25
    alert_var = -0.03

    all_tickers = tickers + ([benchmark] if benchmark else [])
    panel = get_price_panel(all_tickers, start, end=end)

    prices = panel[tickers]
    bench_px = panel[benchmark] if benchmark else None
    latest_table = build_latest_metrics_table(
        prices,
        rf_annual=rf_annual,
        var_level=var_level,
        vol_q_low=vol_q_low,
        vol_q_high=vol_q_high,
        alert_maxdd=alert_maxdd,
        alert_var=alert_var,
        window=window,
        benchmark_prices=bench_px
    )


    fig = make_risk_radar_figure(
        prices,
        latest_table,
        rf_annual=rf_annual,
        window=window,
        corr_lookback=corr_lookback,
        benchmark=benchmark
    )

    save_outputs(fig, latest_table)

if __name__ == "__main__":
    main()
