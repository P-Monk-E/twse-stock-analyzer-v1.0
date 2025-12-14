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
    "1737": "真新",
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "006208": "富邦科技",
}

# ETF 清單（防止股票誤查）
ETF_LIST = {"0050", "0056", "006208"}

def is_etf(code):
    return code in ETF_LIST

def find_ticker_by_name(input_str):
    for t, name in TICKER_NAME_MAP.items():
        if input_str in name:
            return t
    return input_str

def fetch_price_data(code, start, end):
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
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
    return float(sm.OLS(am, X).fit().params["market"])

def calc_alpha(prices_asset, prices_market, rf):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    ar = df.iloc[:, 0].pct_change().dropna()
    mr = df.iloc[:, 1].pct_change().dropna()
    excess_a = ar - rf / 252
    excess_m = mr - rf / 252
    X = sm.add_constant(excess_m)
    return float(sm.OLS(excess_a, X).fit().params["const"]) * 252

def calc_sharpe(prices, rf):
    r = prices.pct_change().dropna()
    return ((r - rf / 252).mean() / r.std()) * np.sqrt(252) if r.std() != 0 else np.nan

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
        except Exception:
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
