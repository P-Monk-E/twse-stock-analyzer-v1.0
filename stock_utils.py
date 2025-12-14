# /mnt/data/stock_utils.py
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 股票 / ETF 名稱對照
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

# ETF 清單
ETF_LIST = {"0050", "0056", "006208"}

def is_etf(code):
    return code in ETF_LIST

def find_ticker_by_name(input_str):
    s = str(input_str).upper()
    for t, name in TICKER_NAME_MAP.items():
        if s in name or s == t:
            return t
    return s  # 保留原本回傳輸入的行為

def fetch_price_data(code, start, end):
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except:
        return None

def calc_beta(prices_asset, prices_market):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    if df.empty:
        return np.nan
    df.columns = ["asset", "market"]
    am = df["asset"].resample("M").last().pct_change().dropna()
    mm = df["market"].resample("M").last().pct_change().dropna()
    if len(am) < 12:
        return np.nan
    X = sm.add_constant(mm)
    return float(sm.OLS(am, X).fit().params.get("market", np.nan))

def calc_alpha(prices_asset, prices_market, rf):
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

def calc_sharpe(prices, rf):
    r = prices.pct_change().dropna()
    if r.std() == 0 or len(r) == 0:
        return np.nan
    return ((r - rf / 252).mean() / r.std()) * np.sqrt(252)

def get_metrics(code, market_close, rf, start, end, is_etf=False):
    df = fetch_price_data(code, start, end)
    if df is None or df.empty:
        return None

    beta = calc_beta(df["Close"], market_close)
    alpha = calc_alpha(df["Close"], market_close, rf)
    sharpe = calc_sharpe(df["Close"], rf)

    if is_etf:
        debt_equity = np.nan
        current_ratio = np.nan
        roe = np.nan
    else:
        try:
            info = yf.Ticker(f"{code}.TW").info
        except:
            info = {}

        total_liab = info.get("totalLiab", np.nan)
        total_assets = info.get("totalAssets", np.nan)

        if np.isnan(total_liab) or np.isnan(total_assets):
            debt_equity = np.nan
        else:
            equity = total_assets - total_liab
            debt_equity = total_liab / equity if equity != 0 else np.nan

        current_ratio = info.get("currentRatio", np.nan)
        roe = info.get("returnOnEquity", np.nan)

    returns = df["Close"].pct_change().dropna()
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
        "df": df
    }
