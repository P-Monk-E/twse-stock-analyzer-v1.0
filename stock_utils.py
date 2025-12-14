from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# -------------------------
# 基本設定 / 名稱映射
# -------------------------
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

ETF_LIST = {"0050", "0056", "006208"}  # 額外白名單；仍支援 00xxxx 模式
_ETF_REGEX = re.compile(r"^00[0-9]{2,4}[A-Z]?$", re.IGNORECASE)

def is_etf(code: str) -> bool:
    if not code:
        return False
    c = str(code).strip().upper()
    return (c in ETF_LIST) or bool(_ETF_REGEX.match(c))

def find_ticker_by_name(input_str: str) -> str:
    s = str(input_str).strip().upper()
    if s in TICKER_NAME_MAP:
        return s
    for t, name in TICKER_NAME_MAP.items():
        if s in name or s in name.upper():
            return t
    return s  # 回傳原字串（可能已是代碼）

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None

# -------------------------
# 財報抓取（健壯版）
# -------------------------

def _first_non_nan(values: pd.Series | pd.Index | list) -> Optional[float]:
    """回傳序列中第一個非 NaN 數字，否則 None。"""
    try:
        s = pd.Series(values, dtype="float64")
        s = s.dropna()
        return float(s.iloc[0]) if not s.empty else None
    except Exception:
        # 有些是 object，需要再嘗試轉型
        try:
            s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
            return float(s.iloc[0]) if not s.empty else None
        except Exception:
            return None

def _find_row(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    """在 df.index 以不分大小寫 '包含' 比對；傳回第一個命中列名。"""
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for p in patterns:
        p = p.lower()
        for low, original in idx_lower.items():
            if p in low:
                return original
    return None

def _get_financial_ratios(code: str) -> tuple[float, float, float]:
    """
    回傳 (debt_equity, current_ratio, roe)
    不可得用 np.nan；盡量容忍多種欄位命名與年/季報差異。
    """
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan

    try:
        t = yf.Ticker(f"{code}.TW")
        # 依序嘗試：年報 -> 季報
        bs_list = [t.balance_sheet, t.quarterly_balance_sheet]
        fs_list = [t.financials, t.quarterly_financials]

        # 先從任一存在的 balance sheet 裡找需要的列
        bs = next((x for x in bs_list if x is not None and not x.empty), None)
        fs = next((x for x in fs_list if x is not None and not x.empty), None)

        # 若兩者皆無，直接返回 NaN
        if bs is None and fs is None:
            return debt_equity, current_ratio, roe

        # --- Equity / Liabilities / Current ---
        # 允許多種別名（Yahoo! 可能更名）
        row_equity = _find_row(bs, [
            "total stockholder equity",
            "total shareholders equity",
            "total equity gross minority interest",
            "total equity",
        ]) if bs is not None else None

        row_assets = _find_row(bs, ["total assets"]) if bs is not None else None
        row_liab   = _find_row(bs, ["total liab", "total liabilities"]) if bs is not None else None

        row_cur_assets = _find_row(bs, ["total current assets", "current assets"]) if bs is not None else None
        row_cur_liab   = _find_row(bs, ["total current liabilities", "current liabilities", "current liab"]) if bs is not None else None

        # 取各列最新一期的值
        def _row_value(df: pd.DataFrame, row_name: Optional[str]) -> Optional[float]:
            if df is None or row_name is None:
                return None
            try:
                # yfinance 的列為 index、欄位為各期；抓第一個非 NaN 值
                return _first_non_nan(df.loc[row_name].values)
            except Exception:
                return None

        v_equity = _row_value(bs, row_equity)
        v_assets = _row_value(bs, row_assets)
        v_liab   = _row_value(bs, row_liab)
        v_cur_a  = _row_value(bs, row_cur_assets)
        v_cur_l  = _row_value(bs, row_cur_liab)

        # 若股東權益找不到，嘗試用 資產 - 負債 估
        if v_equity is None and (v_assets is not None and v_liab is not None):
            v_equity = float(v_assets) - float(v_liab)

        # Debt/Equity
        if v_liab is not None and v_equity is not None and v_equity != 0 and np.isfinite(v_equity):
            debt_equity = float(v_liab) / float(v_equity)

        # Current Ratio
        if v_cur_a is not None and v_cur_l is not None and v_cur_l != 0 and np.isfinite(v_cur_l):
            current_ratio = float(v_cur_a) / float(v_cur_l)

        # --- Net Income（income statement）---
        row_net_income = None
        if fs is not None:
            row_net_income = _find_row(fs, [
                "net income common stockholders",
                "net income applicable to common shares",
                "net income",
                "net income from continuing operations",
            ])
        v_net_income = None
        if fs is not None and row_net_income is not None:
            try:
                v_net_income = _first_non_nan(fs.loc[row_net_income].values)
            except Exception:
                v_net_income = None

        if v_net_income is not None and v_equity is not None and v_equity != 0 and np.isfinite(v_equity):
            roe = float(v_net_income) / float(v_equity)

    except Exception:
        # 抓不到資料即保留 NaN
        pass

    return debt_equity, current_ratio, roe

# -------------------------
# 風險 / 績效計算
# -------------------------
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
    if beta is None or not np.isfinite(beta) or abs(beta) < 1e-6:
        return np.nan
    r = prices.pct_change().dropna()
    if r.empty:
        return np.nan
    ann_return = float(r.mean()) * 252.0
    return (ann_return - rf) / beta

def _calc_madr(prices: pd.Series) -> float:
    r = prices.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(np.abs(r).mean())

# -------------------------
# 主函數
# -------------------------
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

    # —— 核心修復：以健壯抓法取得三個財務比率（僅個股需要） ——
    if not is_etf:
        debt_equity, current_ratio, roe = _get_financial_ratios(code)
    else:
        debt_equity = np.nan
        current_ratio = np.nan
        roe = np.nan

    warnings = {
        "負債權益比": bool(debt_equity <= 1.0) if pd.notna(debt_equity) else False,
        "流動比率": bool(current_ratio >= 1.5) if pd.notna(current_ratio) else False,
        "ROE": bool(roe >= 0.15) if pd.notna(roe) else False,
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
