import pandas as pd
import yfinance as yf

def _fetch_one(ticker: str, start: str, end: str | None = None) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns and df[("Adj Close", ticker)].notna().any():
            px = df[("Adj Close", ticker)].dropna()
        else:
            px = df[("Close", ticker)].dropna()
    else:
        if "Adj Close" in df.columns and df["Adj Close"].notna().any():
            px = df["Adj Close"].dropna()
        else:
            px = df["Close"].dropna()
            
    px.name = ticker
    return px

def get_price_panel(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    series = []
    for t in tickers:
        series.append(_fetch_one(t, start, end))
    prices = pd.concat(series, axis=1).dropna(how="any")
    if prices.shape[0] < 100:
        raise ValueError("Too few rows after cleaning. Either try an earlier start date or use few tickers.")
    return prices


