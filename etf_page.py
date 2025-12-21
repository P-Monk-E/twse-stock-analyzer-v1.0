from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from risk_grading import grade_alpha, grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning
from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import (
    plot_candlestick_with_indicators,
    PLOTLY_TV_CONFIG,
    _ensure_ohlc,
    detect_rsi_divergence,   # ← 新增
)
from watchlist_page import add_to_watchlist


def _sync_symbol_from_input() -> None:
    txt = (st.session_state.get("etf_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]

def _fmt2(x: Optional[float]) -> str:
    try:
        if x is None or pd.isna(x): return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def show(prefill_symbol: Optional[str] = None) -> None:
    st.header("📊 ETF")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入 ETF 代碼或名稱", value=default_symbol, key="etf_symbol", on_change=_sync_symbol_from_input)
    kw = (st.session_state.get("etf_symbol") or "").strip()
    if not kw:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(kw)
        if not is_etf(ticker):
            st.warning("偵測到輸入為個股，請切換至「股票」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt, rf, start, end, is_etf=True)
        if not stats:
            st.warning("查無 ETF 資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")

        # 標題 + 右上角加入觀察
        c1, c2 = st.columns([1, 0.15])
        with c1: st.subheader(f"{name or ticker}（{ticker}）")
        with c2:
            if st.button("＋ 加入觀察", key="btn_watch_etf"):
                add_to_watchlist("etf", ticker, name or ticker)

        # KPI（ETF 含 Treynor + EPS）
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Alpha(年化)", _fmt2(stats.get("Alpha"))); st.caption("越大越好")
        with k2: st.metric("Sharpe Ratio", _fmt2(stats.get("Sharpe Ratio"))); st.caption(">1 佳")
        with k3: st.metric("Treynor Ratio", _fmt2(stats.get("Treynor"))); st.caption("市場單位風險回報")
        with k4: st.metric("Beta", _fmt2(stats.get("Beta"))); st.caption("相對市場波動")
        with k5: st.metric("EPS (TTM)", _fmt2(stats.get("EPS_TTM"))); st.caption("近四次配息合計")

        # 風險摘要 + 分散風險提示
        grades = {
            "Alpha": grade_alpha(stats.get("Alpha")),
            "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
            "Treynor": grade_treynor(stats.get("Treynor")),
        }
        crit, warn, _ = summarize(grades)
        if crit: st.warning("⚠ 風險摘要：" + "、".join(crit) + " 未達標。")
        elif warn: st.info("⚠ 注意：" + "、".join(warn) + " 表現普通。")
        else: st.success("✅ 指標狀態良好。")

        msg = diversification_warning(stats.get("Sharpe Ratio"), stats.get("Treynor"))
        if msg: st.warning(msg)

        # 圖（含 RSI、MACD/KDJ）
        fig = plot_candlestick_with_indicators(_ensure_ohlc(stats["df"]).copy(),
                                               title=f"{name or ticker}（{ticker}）技術圖（日 K）",
                                               uirevision_key=f"{ticker}_1d")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
