from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
from risk_grading import grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_ma

def _sync_symbol_from_input() -> None:
    txt = (st.session_state.get("etf_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]

def _tag(val: Optional[float], thr: float, greater: bool = True) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    return "✅" if ((val >= thr) if greater else (val <= thr)) else "❗"

def _fmt2(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.2f}"
    except Exception:
        return "—"

def show(prefill_symbol: str | None = None) -> None:
    st.header("📊 ETF 專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入 ETF 名稱或代碼（例：0050 / 0056 / 006208 / 00980A）",
                  value=default_symbol, key="etf_symbol", on_change=_sync_symbol_from_input)
    user_input = (st.session_state.get("etf_symbol") or "").strip()
    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
        # 使用強化版 is_etf，00980A 會被正確識別
        if not is_etf(ticker):
            st.warning("偵測到輸入為個股，請切換至「股票」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=True)
        if not stats:
            st.warning("查無 ETF 資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        st.subheader(f"{name or ticker}（{ticker}）")

        # ======= Top KPI：四欄（Treynor 在 Sharpe 右邊）=======
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Alpha(年化)", _fmt2(stats.get("Alpha")))
            st.caption(_tag(stats.get("Alpha"), 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", _fmt2(stats.get("Sharpe Ratio")))
            st.caption(_tag(stats.get("Sharpe Ratio"), 1, True) + " >1 佳")
        with col3:
            st.metric("Treynor Ratio", _fmt2(stats.get("Treynor")))
            st.caption("市場單位風險回報")
        with col4:
            st.metric("Beta", _fmt2(stats.get("Beta")))
            st.caption("相對市場波動")

        # ======= 精簡摘要 =======
        grades = {"Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
                  "Treynor": grade_treynor(stats.get("Treynor"))}
        crit, warn, _ = summarize(grades)
        if crit:
            st.warning("⚠ 風險摘要：**" + "、".join(crit) + "** 未達標。")
        elif warn:
            st.info("⚠ 注意：**" + "、".join(warn) + "** 表現普通。")
        else:
            st.success("✅ 指標狀態良好。")

        # ======= （新增）ETF 專屬系統/非系統性風險提示 =======
        sharpe = stats.get("Sharpe Ratio")
        treynor = stats.get("Treynor")
        non_sys_thr = float(st.session_state.get("non_sys_thr", 0.5))
        sys_thr = float(st.session_state.get("sys_thr", 0.5))
        msg = diversification_warning(sharpe, treynor, non_sys_thr=non_sys_thr, sys_thr=sys_thr)
        if msg:
            st.warning(msg)

        # ======= 圖表 + 波動提示 =======
        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
