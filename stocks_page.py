from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_ma
from risk_grading import (
    grade_sharpe,
    grade_treynor,
    grade_debt_equity,
    grade_current_ratio,
    grade_roe,
    has_any_critical,
)

def _sync_symbol_from_input() -> None:
    txt = (st.session_state.get("stock_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]

def _tag(val: Optional[float], thr: float, greater: bool = True) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    return "✅" if ((val >= thr) if greater else (val <= thr)) else "❗"

def show(prefill_symbol: str | None = None) -> None:
    st.header("📈 股票專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入股票名稱或代碼（例：台積電 或 2330）",
                  value=default_symbol, key="stock_symbol", on_change=_sync_symbol_from_input)
    user_input = (st.session_state.get("stock_symbol") or "").strip()
    if not user_input:
        st.info("請輸入股票名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
        if is_etf(ticker):
            st.warning("偵測到輸入為 ETF，請切換至「ETF」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=False)
        if not stats:
            st.warning("查無該股票資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        st.subheader(f"{name or ticker}（{ticker}）")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        c1, c2, c3 = st.columns(3)
        v = stats.get("負債權益比"); c1.write(f"**負債權益比**：{v if pd.notna(v) else '—'} {_tag(v, 1, False)}")
        v = stats.get("流動比率");   c2.write(f"**流動比率**：{v if pd.notna(v) else '—'} {_tag(v, 1.5, True)}")
        v = stats.get("ROE");       c3.write(f"**ROE**：{(v*100):.2f}% {_tag(v, 0.10, True)}" if pd.notna(v) else "**ROE**：— ❓")

        grades = {}

        g, _ = grade_sharpe(stats["Sharpe Ratio"])
        grades["Sharpe"] = (g, "")
        st.write(f"**Sharpe Ratio**：{stats['Sharpe Ratio']:.2f} {g}")

        g, _ = grade_treynor(stats.get("Treynor"))
        grades["Treynor"] = (g, "")
        st.write(f"**Treynor Ratio**：{stats.get('Treynor', float('nan')):.2f} {g}")

        v = stats["負債權益比"]
        g, _ = grade_debt_equity(v)
        grades["負債權益比"] = (g, "")
        st.write(f"**負債權益比**：{v:.2f} {g}")

        v = stats["流動比率"]
        g, _ = grade_current_ratio(v)
        grades["流動比率"] = (g, "")
        st.write(f"**流動比率**：{v:.2f} {g}")

        v = stats["ROE"]
        g, _ = grade_roe(v)
        grades["ROE"] = (g, "")
        st.write(f"**ROE**：{v*100:.2f}% {g}")

        if has_any_critical(grades):
            st.warning("⚠ 系統警告：至少一項核心風險 / 財務指標未達標")

        
        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")
