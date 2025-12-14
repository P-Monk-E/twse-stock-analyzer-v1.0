from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
from risk_grading import (
    grade_sharpe,
    grade_treynor,
    has_any_critical,
)

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

def show(prefill_symbol: str | None = None) -> None:
    st.header("📊 ETF 專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入 ETF 名稱或代碼（例：0050 / 0056 / 006208）",
                  value=default_symbol, key="etf_symbol", on_change=_sync_symbol_from_input)
    user_input = (st.session_state.get("etf_symbol") or "").strip()
    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", f"{stats.get('Alpha'):.4f}" if stats.get("Alpha") is not None else "—")
            st.caption(_tag(stats.get("Alpha"), 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio'):.2f}" if stats.get("Sharpe Ratio") is not None else "—")
            st.caption(_tag(stats.get("Sharpe Ratio"), 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", f"{stats.get('Beta'):.2f}" if stats.get("Beta") is not None else "—")
            st.caption("相對市場波動")

        grades = {}
        g, _ = grade_sharpe(stats.get("Sharpe Ratio")); grades["Sharpe"] = (g, "")
        st.write(f"**Sharpe Ratio**：{stats.get('Sharpe Ratio', float('nan')):.2f} {g}")

        g, _ = grade_treynor(stats.get("Treynor")); grades["Treynor"] = (g, "")
        st.write(f"**Treynor Ratio**：{stats.get('Treynor', float('nan')):.2f} {g}")

        if has_any_critical(grades):
            st.warning("⚠ 系統警告：至少一項核心風險 / 財務指標未達標")

        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
