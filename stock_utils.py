from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

TICKER_NAME_MAP = {
    "2330": "台積電",
    "2454": "聯發科",
    "2303": "聯電",
    "2618": "長榮航",
    "1737": "臺鹽",
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "006208": "富邦科技",
}
ETF_LIST = {"0050", "0056", "006208"}

def is_etf(code: str) -> bool:
    return code in ETF_LIST

def find_ticker_by_name(input_str: str) -> str:
    s = str(input_str).strip().upper()
    if s in TICKER_NAME_MAP:
        return s
    for t, name in TICKER_NAME_MAP.items():
        if s in name or s in name.upper():
            return t
    return s  # 假設使用者就輸入代碼

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None  # why: 避免 UI 中斷

def _calc_beta(prices_asset: pd.Series, prices_market: pd.Series) -> float | float("nan"):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    if df.empty:
        return np.nan
    df.columns = ["asset", "market"]
    am = df["asset"].resample("M").last().pct_change().dropna()
    mm = df["market"].resample("M").last().pct_change().dropna()
    if len(am) < 12:
        return np.nan
    X = sm.add_constant(mm)
    try:
        return float(sm.OLS(am, X).fit().params.get("market", np.nan))
    except Exception:
        return np.nan

def _calc_alpha(prices_asset: pd.Series, prices_market: pd.Series, rf: float) -> float | float("nan"):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    if df.empty:
        return np.nan
    ar = df.iloc[:, 0].pct_change().dropna()
    mr = df.iloc[:, 1].pct_change().dropna()
    if len(ar) == 0 or len(mr) == 0:
        return np.nan
    excess_a = ar - rf / 252
    excess_m = mr - rf / 252
    X = sm.add_constant(excess_m)
    try:
        return float(sm.OLS(excess_a, X).fit().params.get("const", np.nan)) * 252
    except Exception:
        return np.nan

def _calc_sharpe(prices: pd.Series, rf: float) -> float | float("nan"):
    r = prices.pct_change().dropna()
    if r.std() == 0 or len(r) == 0:
        return np.nan
    return ((r - rf / 252).mean() / r.std()) * np.sqrt(252)

def get_metrics(code: str, market_close: pd.Series, rf: float, start, end, is_etf: bool = False):
    df = fetch_price_data(code, start, end)
    if df is None or df.empty:
        return None

    close = df["Close"]
    beta = _calc_beta(close, market_close)
    alpha = _calc_alpha(close, market_close, rf)
    sharpe = _calc_sharpe(close, rf)

    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    if not is_etf:
        try:
            info = yf.Ticker(f"{code}.TW").info or {}
            tl = info.get("totalLiab", np.nan)
            ta = info.get("totalAssets", np.nan)
            if not (np.isnan(tl) or np.isnan(ta)):
                equity = ta - tl
                debt_equity = (tl / equity) if equity != 0 else np.nan
            current_ratio = info.get("currentRatio", np.nan)
            roe = info.get("returnOnEquity", np.nan)
        except Exception:
            pass  # why: yfinance info 常為 None/timeout

    returns = close.pct_change().dropna()
    madr = np.median(np.abs(returns - returns.mean())) if not returns.empty else np.nan

    return {
        "name": TICKER_NAME_MAP.get(code, ""),
        "負債權益比": debt_equity,
        "流動比率": current_ratio,
        "ROE": roe,
        "Alpha": alpha,
        "Sharpe Ratio": sharpe,
        "Beta": beta,
        "MADR": madr,
        "df": df,
    }
