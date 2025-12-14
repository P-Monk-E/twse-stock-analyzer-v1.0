from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# =========================
# 基本設定
# =========================

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

# =========================
# 工具函數
# =========================

def is_etf(code: str) -> bool:
    return code in ETF_LIST

def find_ticker_by_name(input_str: str) -> str:
    s = str(input_str).strip().upper()
    if s in TICKER_NAME_MAP:
        return s
    for t, name in TICKER_NAME_MAP.items():
        if s in name or s in name.upper():
            return t
    return s

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None

# =========================
# 風險 / 績效計算
# =========================

def _calc_beta(asset: pd.Series, market: pd.Series) -> float:
    df = pd.concat([asset, market], axis=1).dropna()
    if df.empty:
        return np.nan
    df.columns = ["asset", "market"]
    ar = df["asset"].resample("M").last().pct_change().dropna()
    mr = df["market"].resample("M").last().pct_change().dropna()
    if len(ar) < 12:
        return np.nan
    X = sm.add_constant(mr)
    return float(sm.OLS(ar, X).fit().params.get("market", np.nan))

def _calc_alpha(asset: pd.Series, market: pd.Series, rf: float) -> float:
    r = pd.concat([asset, market], axis=1).pct_change().dropna()
    if r.empty:
        return np.nan
    excess_a = r.iloc[:, 0] - rf / 252
    excess_m = r.iloc[:, 1] - rf / 252
    X = sm.add_constant(excess_m)
    return float(sm.OLS(excess_a, X).fit().params.get("const", np.nan)) * 252

def _calc_sharpe(prices: pd.Series, rf: float) -> float:
    r = prices.pct_change().dropna()
    if r.empty or r.std() == 0:
        return np.nan
    return ((r - rf / 252).mean() / r.std()) * np.sqrt(252)

def _calc_treynor(prices: pd.Series, rf: float, beta: float) -> float:
    """
    Treynor = (年化報酬 - rf) / beta；beta ~ 0 時回傳 NaN 避免爆衝。
    """
    if beta is None or not np.isfinite(beta) or abs(beta) < 1e-6:
        return np.nan
    r = prices.pct_change().dropna()
    if r.empty:
        return np.nan
    ann_return = float(r.mean()) * 252.0
    return (ann_return - rf) / beta

def _calc_madr(prices: pd.Series) -> float:
    """
    MADR(Mean Absolute Daily Return)：|日報酬| 的平均，用於衡量日內波動感受。
    """
    r = prices.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(np.abs(r).mean())

# =========================
# 主函數：指標 + 警告
# =========================

def get_metrics(code: str, market_close: pd.Series, rf: float, start, end, is_etf: bool = False):
    df = fetch_price_data(code, start, end)
    if df is None or df.empty:
        return None

    close = df["Close"]

    alpha = _calc_alpha(close, market_close, rf)
    beta = _calc_beta(close, market_close)
    sharpe = _calc_sharpe(close, rf)
    treynor = _calc_treynor(close, rf, beta)
    madr = _calc_madr(close)

    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan

    if not is_etf:
        try:
            t = yf.Ticker(f"{code}.TW")
            bs = t.balance_sheet
            fs = t.financials

            equity = np.nan
            if not bs.empty:
                total_assets = bs.loc["Total Assets"].iloc[0]
                total_liab = bs.loc["Total Liab"].iloc[0]
                current_assets = bs.loc.get("Total Current Assets", [np.nan])[0]
                current_liab = bs.loc.get("Total Current Liabilities", [np.nan])[0]
                equity = total_assets - total_liab
                if np.isfinite(equity) and equity != 0:
                    debt_equity = total_liab / equity
                if np.isfinite(current_liab) and current_liab != 0:
                    current_ratio = current_assets / current_liab
            if not fs.empty and np.isfinite(equity) and equity != 0:
                net_income = fs.loc["Net Income"].iloc[0]
                roe = net_income / equity
        except Exception:
            pass  # 無法取到財報時以 NaN 呈現

    warnings = {
        "負債權益比": bool(debt_equity <= 1.0) if pd.notna(debt_equity) else False,
        "流動比率": bool(current_ratio >= 1.5) if pd.notna(current_ratio) else False,
        "ROE": bool(roe >= 0.15) if pd.notna(roe) else False,
        # 可依需求再擴充：如 MADR > 某閾值判高波動等
    }

    return {
        "name": TICKER_NAME_MAP.get(code, ""),
        "負債權益比": debt_equity,
        "流動比率": current_ratio,
        "ROE": roe,
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": sharpe,
        "Treynor": treynor,
        "MADR": madr,
        "警告": warnings,
        "df": df,
    }
