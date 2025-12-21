# =========================================
# /mnt/data/stocks_page.py
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import (
    plot_candlestick_with_indicators,
    PLOTLY_TV_CONFIG,
    _ensure_ohlc,
    detect_rsi_divergence,   # ← 新增
)
from risk_grading import (
    grade_alpha,
    grade_sharpe,
    grade_debt_equity,
    grade_current_ratio,
    grade_roe,
    summarize,
)
from watchlist_page import add_to_watchlist


def _sync_symbol_from_input() -> None:
    txt = (st.session_state.get("stock_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]


def _fmt2(x: Optional[float]) -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _fmt2pct(x: Optional[float]) -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"


def _fmt_millions(x: Optional[float]) -> str:
    """以百萬（M）顯示，例如 1,317M。"""
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{(float(x) / 1_000_000):,.0f}M"
    except Exception:
        return "—"


def show(prefill_symbol: Optional[str] = None) -> None:
    st.header("📈 股票")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input(
        "輸入股票代碼或名稱",
        value=default_symbol,
        key="stock_symbol",
        on_change=_sync_symbol_from_input,
    )
    kw = (st.session_state.get("stock_symbol") or "").strip()
    if not kw:
        st.info("請輸入股票名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(kw)
        if is_etf(ticker):
            st.warning("偵測到輸入為 ETF，請切換至「ETF」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt, rf, start, end, is_etf=False)
        if not stats:
            st.warning("查無該股票資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")

        # 取值（容錯不同鍵名）
        eps = stats.get("EPS(TTM)", stats.get("EPS_TTM"))
        equity = stats.get("股東權益", stats.get("Equity"))

        # 標題 + 右上角加入觀察
        c1, c2 = st.columns([1, 0.15])
        with c1:
            st.subheader(f"{name or ticker}（{ticker}）")
        with c2:
            if st.button("＋ 加入觀察", key="btn_watch_stock"):
                add_to_watchlist("stock", ticker, name or ticker)

        # ======= KPI 第 1 排（EPS 放最右邊）=======
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Alpha(年化)", _fmt2(stats.get("Alpha")))
            st.caption("越大越好")
        with k2:
            st.metric("Sharpe Ratio", _fmt2(stats.get("Sharpe Ratio")))
            st.caption(">1 佳")
        with k3:
            st.metric("Beta", _fmt2(stats.get("Beta")))
            st.caption("相對市場波動")
        with k4:
            st.metric("EPS (TTM)", _fmt2(eps))
            st.caption("近四季盈餘/股")

        # ======= 風險摘要（不含 Treynor）=======
        grades = {
            "Alpha": grade_alpha(stats.get("Alpha")),
            "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
            "負債權益比": grade_debt_equity(stats.get("負債權益比")),
            "流動比率": grade_current_ratio(stats.get("流動比率")),
            "ROE": grade_roe(stats.get("ROE")),
        }
        crit, warn, _ = summarize(grades)
        if crit:
            st.warning("⚠ 風險摘要：" + "、".join(crit) + " 未達標。")
        elif warn:
            st.info("⚠ 注意：" + "、".join(warn) + " 表現普通。")
        else:
            st.success("✅ 指標狀態良好。")

        # ======= 財務列（股東權益以百萬顯示；EPS 已移至上排）=======
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("負債權益比", _fmt2(stats.get("負債權益比")))
        with c2:
            st.metric("流動比率", _fmt2(stats.get("流動比率")))
        with c3:
            st.metric("ROE", _fmt2pct(stats.get("ROE")))
        with c4:
            st.metric("股東權益", _fmt_millions(equity))

        # ======= 圖（含 RSI、MACD/KDJ；線條皆連續實線）=======
        fig = plot_candlestick_with_indicators(
            _ensure_ohlc(stats["df"]).copy(),
            title=f"{name or ticker}（{ticker}）技術圖（日 K）",
            uirevision_key=f"{ticker}_1d",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")
