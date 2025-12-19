# =========================================
# /mnt/data/stocks_page.py
# 修正：資料慢一天 → end(台北時區)=今天+2天 + 以 7d 補齊；休市日不顯示；60m/日切換
# =========================================
from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import pytz

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG
from risk_grading import (
    grade_alpha, grade_sharpe, grade_treynor,
    grade_debt_equity, grade_current_ratio, grade_roe, summarize,
)

def _fmt2(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.2f}"
def _fmt2pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x*100:.2f}%"
def _fmt0(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:,.0f}"

def _normalize_tw_ticker(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW", ".TWO")) or s.startswith("^") else f"{s}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_60m(ticker: str, period: str = "90d") -> pd.DataFrame:
    try:
        df = yf.Ticker(_normalize_tw_ticker(ticker)).history(period=period, interval="60m", auto_adjust=False)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close"]].dropna(how="any")
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def _market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end, auto_adjust=False)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy(); s.name = idx; return s
        except Exception:
            continue
    return None

def _prepare_tf_df(ticker: str, daily_df: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, str]:
    if tf == "60m":
        return _download_ohlc_60m(ticker, "90d"), "（60 分鐘）"
    return daily_df.copy(), "（日 K）"

def _kpi_grid(items: list[tuple[str, str, str]], cols: int = 5) -> None:
    if not items: return
    it = iter(items)
    for _ in range((len(items) + cols - 1) // cols):
        cs = st.columns(cols)
        for c in cs:
            try:
                name, val, hp = next(it)
            except StopIteration:
                break
            with c:
                st.metric(label=name, value=val, help=hp or None)

def _backfill_latest_daily(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """若少了最近交易日，用 7d 歷史補齊尾端。"""
    try:
        tail = yf.Ticker(_normalize_tw_ticker(ticker)).history(period="7d", interval="1d", auto_adjust=False)
        if tail is None or tail.empty:
            return df
        tail = tail[["Open", "High", "Low", "Close"]].dropna(how="any")
        out = pd.concat([df[["Open", "High", "Low", "Close"]], tail]).~drop_duplicates(subset=None)  # placeholder
    except Exception:
        return df
