from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# ---------- 基本設定 ----------
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

# ---------- 財報工具 ----------
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

def _get_shares_outstanding(t: yf.Ticker) -> Optional[float]:
    """盡量取得流通在外股數；失敗回 None。"""
    # a) 全歷史 shares（新 yfinance 提供）
    try:
        sf = t.get_shares_full(start="2000-01-01", end=None)
        if sf is not None and not sf.empty:
            val = float(sf.dropna().iloc[-1])
            if val > 0:
                return val
    except Exception:
        pass
    # b) fast_info
    try:
        val = t.fast_info.get("sharesOutstanding")
        if val and float(val) > 0:
            return float(val)
    except Exception:
        pass
    # c) info
    try:
        val = t.info.get("sharesOutstanding")
        if val and float(val) > 0:
            return float(val)
    except Exception:
        pass
    return None

def _get_financials(code: str) -> Tuple[float, float, float, float, float]:
    """
    回傳 (debt_equity, current_ratio, roe, equity, eps_ttm)；不可得則為 np.nan。
    """
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    equity_val = np.nan
    eps_ttm = np.nan

    try:
        t = yf.Ticker(f"{code}.TW")

        # 年/季報
        bs_year = t.balance_sheet
        bs_quarter = t.quarterly_balance_sheet
        fs_year = t.financials
        fs_quarter = t.quarterly_financials

        bs = next((x for x in [bs_year, bs_quarter] if x is not None and not x.empty), None)
        fs = next((x for x in [fs_year, fs_quarter] if x is not None and not x.empty), None)

        # ---- Equity / Liabilities / Current ----
        row_equity = _find_row(bs, [
            "total stockholder equity",
            "total shareholders equity",
            "total equity gross minority interest",
            "total equity",
        ]) if bs is not None else None
        row_assets = _find_row(bs, ["total assets"]) if bs is not None else None
        row_liab   = _find_row(bs, ["total liab", "total liabilities"]) if bs is not None else None
        row_cur_a  = _find_row(bs, ["total current assets", "current assets"]) if bs is not None else None
        row_cur_l  = _find_row(bs, ["total current liabilities", "current liabilities", "current liab"]) if bs is not None else None

        v_equity = _latest_from_row(bs, row_equity)
        v_assets = _latest_from_row(bs, row_assets)
        v_liab   = _latest_from_row(bs, row_liab)
        v_ca     = _latest_from_row(bs, row_cur_a)
        v_cl     = _latest_from_row(bs, row_cur_l)

        if v_equity is None and (v_assets is not None and v_liab is not None):
            v_equity = float(v_assets) - float(v_liab)
        if v_equity is not None and np.isfinite(v_equity):
            equity_val = float(v_equity)

        if v_liab is not None and v_equity is not None and v_equity != 0 and np.isfinite(v_equity):
            debt_equity = float(v_liab) / float(v_equity)
        if v_ca is not None and v_cl is not None and v_cl != 0 and np.isfinite(v_cl):
            current_ratio = float(v_ca) / float(v_cl)

        # ---- ROE / EPS(TTM) ----
        # 淨利：年報（最近一期）與季報（近四期合計）
        v_net_income_yr = None
        if fs_year is not None and not fs_year.empty:
            row_ni_yr = _find_row(fs_year, [
                "net income common stockholders",
                "net income applicable to common shares",
                "net income",
                "net income from continuing operations",
            ])
            v_net_income_yr = _latest_from_row(fs_year, row_ni_yr)

        v_net_income_ttm = None
        if fs_quarter is not None and not fs_quarter.empty:
            row_ni_q = _find_row(fs_quarter, [
                "net income common stockholders",
                "net income applicable to common shares",
                "net income",
                "net income from continuing operations",
            ])
            if row_ni_q is not None and row_ni_q in fs_quarter.index:
                try:
                    qvals = pd.to_numeric(fs_quarter.loc[row_ni_q], errors="coerce").dropna()
                    if not qvals.empty:
                        v_net_income_ttm = float(qvals.sort_index(ascending=True).tail(4).sum())
                except Exception:
                    pass

        # ROE
        if v_net_income_yr is not None and v_equity is not None and v_equity != 0 and np.isfinite(v_equity):
            roe = float(v_net_income_yr) / float(v_equity)

        # EPS(TTM) = 近四季淨利 / 流通在外股數
        shares = _get_shares_outstanding(t)
        if v_net_income_ttm is not None and shares is not None and shares > 0:
            eps_ttm = float(v_net_income_ttm) / float(shares)

    except Exception:
        # 任何錯誤都回 NaN；不拋出，以確保頁面穩定
        pass

    return debt_equity, current_ratio, roe, equity_val, eps_ttm

# ---------- 風險 / 績效計算 ----------
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

# ---------- 主函數 ----------
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

    if not is_etf:
        debt_equity, current_ratio, roe, equity_val, eps_ttm = _get_financials(code)
    else:
        debt_equity = current_ratio = roe = equity_val = eps_ttm = np.nan

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
        "Equity": equity_val,      # 新增
        "EPS_TTM": eps_ttm,        # 新增
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": sharpe,
        "Treynor": treynor,
        "MADR": madr,
        "警告": warnings,
        "df": df,
    }
