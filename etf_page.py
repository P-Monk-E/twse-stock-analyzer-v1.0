from __future__ import annotations
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import pytz

from risk_grading import grade_alpha, grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning
from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG, _ensure_ohlc

def _normalize_tw_ticker(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW",".TWO")) or s.startswith("^") else f"{s}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_60m(ticker: str, period: str="90d") -> pd.DataFrame:
    try:
        df = yf.Ticker(_normalize_tw_ticker(ticker)).history(period=period, interval="60m", auto_adjust=False)
        return _ensure_ohlc(df)
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
    if tf == "60m": return _download_ohlc_60m(ticker, "90d"), "（60 分鐘）"
    return daily_df.copy(), "（日 K）"

def _backfill_latest_daily(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        tail = yf.Ticker(_normalize_tw_ticker(ticker)).history(period="7d", interval="1d", auto_adjust=False)
        tail = _ensure_ohlc(tail)
        out = pd.concat([_ensure_ohlc(df), tail])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out
    except Exception:
        return _ensure_ohlc(df)

def _tpe_time_range(days: int=366) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = pytz.timezone("Asia/Taipei")
    now_tpe = pd.Timestamp.now(tz=tz)
    end_aware = now_tpe.normalize() + pd.Timedelta(days=2)
    start_aware = end_aware - pd.Timedelta(days=days)
    return start_aware.tz_convert(None), end_aware.tz_convert(None)

def _kpi_grid(items: list[tuple[str,str,str]], cols: int=4) -> None:
    if not items: return
    it = iter(items)
    for _ in range((len(items)+cols-1)//cols):
        cs = st.columns(cols)
        for c in cs:
            try: n,v,h = next(it)
            except StopIteration: break
            with c: st.metric(label=n, value=v, help=h or None)

def render(prefill_symbol: Optional[str]=None) -> None:
    st.header("ETF")
    c1, c2 = st.columns([3,2])
    with c1:
        default_kw = prefill_symbol or st.session_state.get("last_etf_kw","0050")
        keyword = st.text_input("輸入 ETF 代碼或名稱", value=default_kw)
    with c2:
        tf = st.radio("K 線週期", options=["60m","日"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：0050 或 台灣50）"); return

    try:
        ticker = find_ticker_by_name(keyword)
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_etf_kw"] = keyword

        if not is_etf(ticker):
            st.warning("這不是 ETF，請改到「股票」分頁查詢。"); return

        start, end = _tpe_time_range(366)
        market_close = _market_close_series(start, end)
        if market_close is None:
            st.error("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）"); return
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=True)
        if not stats: st.error("查無此 ETF 的歷史資料。"); return

        base_df = _ensure_ohlc(stats["df"])
        base_df = _backfill_latest_daily(ticker, base_df)

        with st.container(border=True):
            st.subheader(f"{name or ticker}（{ticker}）")
            div_warn = diversification_warning(ticker)
            if div_warn: st.warning(div_warn)
            grades = {"Sharpe": grade_sharpe(stats.get("Sharpe Ratio")), "Treynor": grade_treynor(stats.get("Treynor")), "Alpha": grade_alpha(stats.get("Alpha"))}
            crit, warn, good = summarize(grades)
            if crit: st.error("關鍵風險：" + "、".join(crit))
            if warn: st.warning("注意項：" + "、".join(warn))
            if good: st.success("達標：" + "、".join(good))

            _kpi_grid([
                ("Alpha", f"{stats.get('Alpha', float('nan')):.2f}", ""),
                ("Beta", f"{stats.get('Beta', float('nan')):.2f}", ""),
                ("Sharpe", f"{stats.get('Sharpe Ratio', float('nan')):.2f}", ""),
                ("Treynor", f"{stats.get('Treynor', float('nan')):.2f}", ""),
            ], cols=4)

        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf)
        if tf_df.empty: st.error("查無對應週期的價格資料。"); return

        title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
        fig = plot_candlestick_with_indicators(tf_df, title=title, uirevision_key=f"{ticker}_{tf}")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")

def show(prefill_symbol: Optional[str]=None) -> None:
    render(prefill_symbol=prefill_symbol)
