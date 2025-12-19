# =========================================
# /mnt/data/stocks_page.py
# 60m/日 K 切換；主圖含 BB；中圖 RSI；下圖 MACD 柱體 + KDJ(J)
# 嚴格分流（只允許個股）；KPI 區完整；相容 app.py 的 show()
# =========================================
from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG
from risk_grading import (
    grade_alpha, grade_sharpe, grade_treynor,
    grade_debt_equity, grade_current_ratio, grade_roe, summarize,
)

# ---------- 格式 ----------
def _fmt2(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.2f}"
def _fmt2pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x*100:.2f}%"
def _fmt0(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:,.0f}"

# ---------- 市場 / 價格 ----------
def _normalize_tw_ticker_once(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW", ".TWO")) or s.startswith("^") else f"{s}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_intraday(ticker: str, interval: str = "60m", period: str = "90d") -> pd.DataFrame:
    try:
        df = yf.Ticker(_normalize_tw_ticker_once(ticker)).history(period=period, interval=interval)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close"]].dropna(how="any")
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy(); s.name = idx; return s
        except Exception:
            continue
    return None

def _prepare_tf_df(ticker: str, base_daily_df: pd.DataFrame, tf_label: str) -> Tuple[pd.DataFrame, str]:
    if tf_label == "60m":
        return _download_ohlc_intraday(ticker, "60m", "90d"), "（60 分鐘）"
    else:
        return base_daily_df.copy(), "（日 K）"

def _kpi_grid(metrics: list[tuple[str, str, str]], cols: int = 5) -> None:
    if not metrics: return
    rows = (len(metrics) + cols - 1) // cols
    it = iter(metrics)
    for _ in range(rows):
        cs = st.columns(cols)
        for c in cs:
            try: name, val, hp = next(it)
            except StopIteration: break
            with c: st.metric(label=name, value=val, help=hp or None)

# ---------- 主頁面 ----------
def render(prefill_symbol: Optional[str] = None) -> None:
    st.header("股票")
    c1, c2 = st.columns([3, 2])
    with c1:
        default_kw = prefill_symbol or st.session_state.get("last_stock_kw", "2330")
        keyword = st.text_input("輸入股票代碼或名稱", value=default_kw)
    with c2:
        tf_label = st.radio("K 線週期", options=["60m", "日"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：2330 或 台積電）"); return
    try:
        ticker = find_ticker_by_name(keyword)
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_stock_kw"] = keyword

        if is_etf(ticker):
            st.warning("這是 ETF，請改至「ETF」分頁查詢。"); return

        end = pd.Timestamp.today().normalize(); start = end - pd.Timedelta(days=365)
        market_close = _get_market_close_series(start, end)
        if market_close is None:
            st.error("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）"); return
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=False)
        if not stats: st.error("查無此標的的歷史資料。"); return

        with st.container(border=True):
            st.subheader(f"{name or ticker}（{ticker}）")
            grades = {
                "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
                "Treynor": grade_treynor(stats.get("Treynor")),
                "Alpha": grade_alpha(stats.get("Alpha")),
                "負債權益比": grade_debt_equity(stats.get("負債權益比")),
                "流動比率": grade_current_ratio(stats.get("流動比率")),
                "ROE": grade_roe(stats.get("ROE")),
            }
            crit, warn, good = summarize(grades)
            if crit: st.error("關鍵風險：" + "、".join(crit))
            if warn: st.warning("注意項：" + "、".join(warn))
            if good: st.success("達標：" + "、".join(good))

            kpis = [
                ("Alpha", _fmt2(stats.get("Alpha")), "相對市場超額報酬（年化）"),
                ("Beta", _fmt2(stats.get("Beta")), "相對市場波動敏感度"),
                ("Sharpe", _fmt2(stats.get("Sharpe Ratio")), "風險調整後報酬"),
                ("Treynor", _fmt2(stats.get("Treynor")), "以 Beta 計的風險調整報酬"),
                ("MADR", _fmt2(stats.get("MADR")), "最大向下偏離幅度"),
                ("負債權益比", _fmt2(stats.get("負債權益比")), ""),
                ("流動比率", _fmt2(stats.get("流動比率")), ""),
                ("ROE", _fmt2pct(stats.get("ROE")), ""),
                ("股東權益", _fmt0(stats.get("Equity")), ""),
                ("EPS(TTM)", _fmt2(stats.get("EPS_TTM")), "近四季合計"),
            ]
            _kpi_grid(kpis, cols=5)

        base_df: pd.DataFrame = stats["df"].copy()
        if not isinstance(base_df.index, pd.DatetimeIndex): base_df.index = pd.to_datetime(base_df.index)
        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf_label)
        if tf_df.empty:
            st.error("查無對應週期的價格資料。"); return

        title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
        fig = plot_candlestick_with_indicators(tf_df, title=title, uirevision_key=f"{ticker}_{tf_label}")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)
    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")

def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
