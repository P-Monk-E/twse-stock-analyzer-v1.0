# =========================================
# /mnt/data/stock_utils.py
# =========================================
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

DEFAULT_RF = 0.012
DEFAULT_YEARS = 3

TICKER_NAME_MAP: Dict[str, str] = {
    "2330": "台積電", "2454": "聯發科", "2303": "聯電", "2317": "鴻海",
    "2881": "富邦金", "2882": "國泰金", "2603": "長榮", "2618": "長榮航",
    "1216": "統一", "2383": "台光電", "2313": "華通", "2211": "長榮鋼",
    "2451": "創見", "2377": "微星", "2379": "瑞昱", "1303": "南亞", "2615": "萬海",
    "0050": "元大台灣50", "0056": "元大高股息", "006208": "富邦台50", "00980A": "台新永續高息",
}

_ETF_RX = re.compile(r"^(00\d{2,}|009\d{2}[A-Z]?)$")

def is_etf(code: str) -> bool:
    return bool(_ETF_RX.match(str(code).upper()))

def find_ticker_by_name(q: str) -> Optional[str]:
    s = str(q).strip()
    if not s:
        return None
    if s.isdigit() or s.upper().endswith((".TW", ".TWO")):
        return s.upper().removesuffix(".TW").removesuffix(".TWO")
    for k, v in TICKER_NAME_MAP.items():
        if v == s:
            return k
    return None

# -------- price fetch with robust fallbacks --------
def _history_for(ticker: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame | None:
    t = yf.Ticker(ticker)
    # 1) start/end
    try:
        if start and end:
            df = t.history(start=start, end=end, interval="1d")
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    # 2) period fallback（由長到短再到 max）
    for per in ["3y", "2y", "1y", "6mo", "max"]:
        try:
            df = t.history(period=per, interval="1d")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

def fetch_price_data(code: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame | None:
    cands = [code] if code.endswith((".TW", ".TWO")) else [f"{code}.TW", f"{code}.TWO"]
    for c in cands:
        df = _history_for(c, start, end)
        if df is not None and not df.empty:
            return df
    return None

def _fetch_market_close(start: Optional[datetime], end: Optional[datetime]) -> Optional[pd.Series]:
    try:
        t = yf.Ticker("^TWII")
        if start and end:
            df = t.history(start=start, end=end, interval="1d")
            if df is not None and not df.empty:
                return df["Close"]
        for per in ["3y", "2y", "1y", "6mo", "max"]:
            df = t.history(period=per, interval="1d")
            if df is not None and not df.empty:
                return df["Close"]
    except Exception:
        pass
    return None

# -------- KPIs --------
def _calc_alpha(close: pd.Series, market_close: pd.Series, rf: float) -> float:
    try:
        df = pd.concat([close.pct_change().rename("y"),
                        market_close.pct_change().rename("x")], axis=1).dropna()
        if df.empty:
            return np.nan
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"] - rf / 252, X).fit()
        a_daily = model.params["const"]
        return float((1 + a_daily) ** 252 - 1)
    except Exception:
        return np.nan

def _calc_beta(close: pd.Series, market_close: pd.Series) -> float:
    try:
        df = pd.concat([close.pct_change().rename("y"),
                        market_close.pct_change().rename("x")], axis=1).dropna()
        if df.empty:
            return np.nan
        cov = np.cov(df["y"], df["x"])
        if np.isnan(cov).any() or cov.shape != (2, 2):
            return np.nan
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return np.nan

def _calc_sharpe(close: pd.Series, rf: float) -> float:
    try:
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        excess = r - rf / 252
        sr = excess.mean() / excess.std()
        return float(sr * np.sqrt(252))
    except Exception:
        return np.nan

def _calc_treynor(close: pd.Series, rf: float, beta: float) -> float:
    try:
        if beta is None or np.isnan(beta) or beta == 0:
            return np.nan
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        excess = r - rf / 252
        er = float((1 + excess.mean()) ** 252 - 1)
        return float(er / beta)
    except Exception:
        return np.nan

def _calc_madr(close: pd.Series) -> float:
    try:
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        return float(np.abs(r).mean())
    except Exception:
        return np.nan

# -------- fundamentals (best-effort) --------
def _pick_ticker_for_fundamentals(code: str) -> yf.Ticker:
    cands = [code] if code.endswith((".TW", ".TWO")) else [f"{code}.TW", f"{code}.TWO"]
    for c in cands:
        try:
            t = yf.Ticker(c)
            _ = t.fast_info
            return t
        except Exception:
            try:
                if not yf.Ticker(c).history(period="1mo").empty:
                    return yf.Ticker(c)
            except Exception:
                pass
    return yf.Ticker(f"{code}.TW")

# -------- public API --------
def get_metrics(
    code: str,
    market_close: Optional[pd.Series],
    rf: Optional[float],
    start: Optional[datetime],
    end: Optional[datetime],
    is_etf: bool = False,
) -> Optional[Dict[str, Any]]:
    # Auto-fill duration, rf, market
    if start is None or end is None:
        today = datetime.now().date()
        start = datetime.combine(today - timedelta(days=365 * DEFAULT_YEARS), datetime.min.time())
        end = datetime.combine(today, datetime.min.time())
    if rf is None:
        rf = DEFAULT_RF
    if market_close is None:
        market_close = _fetch_market_close(start, end)

    df = fetch_price_data(code, start, end)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None

    alpha = _calc_alpha(close, market_close, rf) if market_close is not None else np.nan
    beta = _calc_beta(close, market_close) if market_close is not None else np.nan
    sharpe = _calc_sharpe(close, rf)
    treynor = _calc_treynor(close, rf, beta)
    madr = _calc_madr(close)

    # fundamentals (best-effort; optional)
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    equity_val = np.nan
    eps_ttm = np.nan
    try:
        t = _pick_ticker_for_fundamentals(code)
        bs_y = t.balance_sheet
        bs_q = t.quarterly_balance_sheet
        fs_q = t.quarterly_financials

        # Current ratio
        try:
            ca_row = [i for i in bs_q.index if re.search(r"(?i)total current assets|流動資產", str(i))]
            cl_row = [i for i in bs_q.index if re.search(r"(?i)total current liab|流動負債", str(i))]
            if ca_row and cl_row:
                ca = float(pd.to_numeric(bs_q.loc[ca_row[0]].iloc[0], errors="coerce"))
                cl = float(pd.to_numeric(bs_q.loc[cl_row[0]].iloc[0], errors="coerce"))
                if cl:
                    current_ratio = ca / cl
        except Exception:
            pass

        # ROE (TTM / last equity)
        try:
            ni_row = [i for i in fs_q.index if re.search(r"(?i)net income|淨利", str(i))]
            eq_row = [i for i in bs_q.index if re.search(r"(?i)total stockholder|股東權益", str(i))]
            if ni_row and eq_row:
                ni_ttm = float(pd.to_numeric(fs_q.loc[ni_row[0]].iloc[:4], errors="coerce").sum())
                eq_last = float(pd.to_numeric(bs_q.loc[eq_row[0]].iloc[0], errors="coerce"))
                if eq_last:
                    roe = ni_ttm / eq_last
        except Exception:
            pass

        # Equity (annual)
        try:
            er = [i for i in bs_y.index if re.search(r"(?i)total stockholder|股東權益", str(i))]
            if er:
                equity_val = float(pd.to_numeric(bs_y.loc[er[0]].iloc[0], errors="coerce"))
        except Exception:
            pass

        # EPS TTM
        try:
            if is_etf:
                dv = t.dividends
                eps_ttm = float(dv.iloc[-4:].sum()) if dv is not None and not dv.empty else np.nan
            else:
                shares = t.get_shares_full(start=start, end=end)
                if shares is not None and not shares.empty:
                    shares_last = float(shares.iloc[-1])
                    ni_row = [i for i in fs_q.index if re.search(r"(?i)net income|淨利", str(i))]
                    if ni_row and shares_last:
                        net_ttm = float(pd.to_numeric(fs_q.loc[ni_row[0]].iloc[:4], errors="coerce").sum())
                        eps_ttm = net_ttm / shares_last
        except Exception:
            pass
    except Exception:
        pass

    return {
        "name": TICKER_NAME_MAP.get(code, ""),
        "負債權益比": debt_equity,
        "流動比率": current_ratio,
        "ROE": roe,
        "Equity": equity_val,
        "EPS_TTM": eps_ttm,
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe": sharpe,      # ← 與頁面鍵一致
        "Treynor": treynor,
        "MADR": madr,
        "df": df,
    }
