from __future__ import annotations

import re
from typing import Optional, Tuple

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

ETF_LIST = {"0050", "0056", "006208"}
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
    return s

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None

# -------------------------
# 財報抓取（健壯版）
# -------------------------
def _first_non_nan(values) -> Optional[float]:
    try:
        s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else None
    except Exception:
        return None

def _find_row(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for p in patterns:
        p = p.lower()
        for low, original in idx_lower.items():
            if p in low:
                return original
    return None

def _latest_from_row(df: pd.DataFrame, row_name: Optional[str]) -> Optional[float]:
    if df is None or row_name is None:
        return None
    try:
        return _first_non_nan(df.loc[row_name].values)
    except Exception:
        return None

def _get_financial_ratios(code: str) -> Tuple[float, float, float, float, float]:
    """
    回傳 (debt_equity, current_ratio, roe, equity, eps_ttm)
    任何不可得回 np.nan。
    """
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    equity_val = np.nan
    eps_ttm = np.nan

    try:
        t = yf.Ticker(f"{code}.TW")
        # 年/季報
        bs = next((x for x in [t.balance_sheet, t.quarterly_balance_sheet] if x is not None and not x.empty), None)
        fs = next((x for x in [t.financials, t.quarterly_financials] if x is not None and not x.empty), None)

        # ------ Equity / Liabilities ------
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

        v_equity = _latest_from_row(bs, row_equity)
        v_assets = _latest_from_row(bs, row_assets)
        v_liab   = _latest_from_row(bs, row_liab)
        v_cur_a  = _latest_from_row(bs, row_cur_assets)
        v_cur_l  = _latest_from_row(bs, row_cur_liab)

        if v_equity is None and (v_assets is not None and v_liab is not None):
            v_equity = float(v_assets) - float(v_liab)

        if v_equity is not None and np.isfinite(v_equity):
            equity_val = float(v_equity)

        if v_liab is not None and v_equity is not None and v_equity != 0 and np.isfinite(v_equity):
            debt_equity = float(v_liab) / float(v_equity)

        if v_cur_a is not None and v_cur_l is not None and v_cur_l != 0 and np.isfinite(v_cur_l):
            current_ratio = float(v_cur_a) / float(v_cur_l)

        # ------ ROE / EPS(TTM) ------
        v_net_income_annual = None
        v_net_income_qsum = None
        if fs is not None:
            row_net_income = _find_row(fs, [
                "net income common stockholders",
                "net income applicable to common shares",
                "net income",
                "net income from continuing operations",
            ])
            if row_net_income is not None:
                # 年報第一期
                v_net_income_annual = _latest_from_row(t.financials, row_net_income) if (t.financials is not None and not t.financials.empty) else None
                # 近四季合計
                try:
                    q = t.quarterly_financials
                    if q is not None and not q.empty and row_net_income in q.index:
                        vals = pd
