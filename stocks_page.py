# =========================================
# /mnt/data/stocks_page.py
# =========================================
from __future__ import annotations
import math
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import pytz

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG, _ensure_ohlc
from risk_grading import (
    grade_alpha, grade_sharpe, grade_treynor,
    grade_debt_equity, grade_current_ratio, grade_roe, summarize,
)

def _fmt2(x: Optional[float]) -> str:  return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.2f}"
def _fmt2pct(x: Optional[float]) -> str: return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x*100:.2f}%"
def _fmt0(x: Optional[float]) -> str:  return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:,.0f}"

def _normalize_tw_ticker(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW",".TWO")) or s.startswith("^") else f"{s}.TW"

def _to_tpe_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None: idx = idx.tz_localize("UTC")
    return idx.tz_convert("Asia/Taipei").tz_localize(None)

def _resample_to_60m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex): return pd.DataFrame(columns=["Open","High","Low","Close"])
    out = df.resample("60min", label="right", closed="right").agg({"Open":"first","High":"max","Low":"min","Close":"last"})
    return out.dropna(how="any")

def _sanitize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    x = x.between_time("09:00","13:59")  # why: 只留交易時段
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_60m(ticker: str, period: str="60d") -> pd.DataFrame:
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy(); df.index = _to_tpe_naive(df.index)
        df = _ensure_ohlc(df)
        return _sanitize_intraday(df)

    try:
        t = yf.Ticker(_normalize_tw_ticker(ticker))

        df = _clean(t.history(period=period, interval="60m", auto_adjust=False))
        if not df.empty: return df

        df30 = _clean(t.history(period=period, interval="30m", auto_adjust=False))
        if not df30.empty: return _resample_to_60m(df30)

        df15 = _clean(t.history(period=period, interval="15m", auto_adjust=False))
        if not df15.empty: return _resample_to_60m(df15)

        return pd.DataFrame(columns=["Open","High","Low","Close"])
    except Exception:
        return pd.DataFrame(columns=["Open","High","Low","Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def _market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    for idx in ["^TWII","^TAIEX","^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end, auto_adjust=False)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy(); s.name = idx; return s
        except Exception:
            continue
    return None

def _prepare_tf_df(ticker: str, daily_df: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, str]:
    if tf == "60m": return _download_ohlc_60m(ticker, "60d"), "（60 分鐘）"
    return daily_df.copy(), "（日 K）"

def _backfill_latest_daily(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        tail = yf.Ticker(_normalize_tw_ticker(ticker)).history(period="7d", interval="1d", auto_adjust=False)
        tail = _ensure_ohlc(tail)
        out  = pd.concat([_ensure_ohlc(df), tail])
        out  = out[~out.index.duplicated(keep="last")].sort_index()
        return out
    except Exception:
        return _ensure_ohlc(df)

def _tpe_time_range(days: int=366) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = pytz.timezone("Asia/Taipei")
    now_tpe = pd.Timestamp.now(tz=tz)
    end_aware   = now_tpe.normalize() + pd.Timedelta(days=2)
    start_aware = end_aware - pd.Timedelta(days=days)
    return start_aware.tz_convert(None), end_aware.tz_convert(None)

def _kpi_grid(items: list[tuple[str,str,str]], cols: int=5) -> None:
    if not items: return
    it = iter(items)
    for _ in range((len(items)+cols-1)//cols):
        cs = st.columns(cols)
        for c in cs:
            try: n,v,h = next(it)
            except StopIteration: break
            with c: st.metric(label=n, value=v, help=h or None)

def render(prefill_symbol: Optional[str]=None) -> None:
    st.header("股票")
    c1, c2 = st.columns([3,2])
    with c1:
        default_kw = prefill_symbol or st.session_state.get("last_stock_kw","2330")
        keyword = st.text_input("輸入股票代碼或名稱", value=default_kw)
    with c2:
        tf = st.radio("K 線週期", options=["60m","日"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：2330 或 台積電）"); return

    try:
        ticker = find_ticker_by_name(keyword)
        name   = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_stock_kw"] = keyword

        if is_etf(ticker):
            st.warning("這是 ETF，請改至「ETF」分頁查詢。"); return

        start, end = _tpe_time_range(366)
        market_close = _market_close_series(start, end)
        if market_close is None:
            st.error("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）"); return
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=False)
        if not stats: st.error("查無此標的的歷史資料。"); return

        base_df = _ensure_ohlc(stats["df"])
        base_df = _backfill_latest_daily(ticker, base_df)

        with st.container(border=True):
            st.subheader(f"{name or ticker}（{ticker}）")

            # KPI 數字（股票不顯示 Treynor）
            _kpi_grid(
                [
                    ("Sharpe", _fmt2(stats.get("Sharpe Ratio")), "報酬/波動"),
                    ("Alpha",  _fmt2(stats.get("Alpha")), "風險調整超額"),
                    ("負債權益比", _fmt2(stats.get("負債權益比")), ""),
                    ("流動比率",   _fmt2(stats.get("流動比率")), ""),
                    ("ROE",     _fmt2pct(stats.get("ROE")), ""),
                    ("股東權益", _fmt0(stats.get("股東權益")), ""),
                    ("EPS(TTM)", _fmt2(stats.get("EPS(TTM)")), ""),
                ],
                cols=7,
            )

            grades = {
                "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
                "Treynor": grade_treynor(None),  # why: 股票不評 Treynor，給 None 即不加權
                "Alpha":  grade_alpha(stats.get("Alpha")),
                "負債權益比": grade_debt_equity(stats.get("負債權益比")),
                "流動比率":   grade_current_ratio(stats.get("流動比率")),
                "ROE":      grade_roe(stats.get("ROE")),
            }
            crit, warn, good = summarize(grades)
            if crit: st.error("關鍵風險：" + "、".join(crit))
            if warn: st.warning("注意項：" + "、".join(warn))
            if good: st.success("達標：" + "、".join(good))

        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf)
        if tf_df.empty: st.error("查無對應週期的價格資料。"); return

        title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
        fig = plot_candlestick_with_indicators(tf_df, title=title, uirevision_key=f"{ticker}_{tf}")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")

def show(prefill_symbol: Optional[str]=None) -> None:
    render(prefill_symbol=prefill_symbol)
