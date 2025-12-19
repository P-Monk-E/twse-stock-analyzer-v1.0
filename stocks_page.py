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
    grade_alpha,
    grade_sharpe,
    grade_debt_equity,
    grade_current_ratio,
    grade_roe,
    summarize,
)
from watchlist_page import add_to_watchlist  # 直接寫入觀察名單  【函式介面】:contentReference[oaicite:3]{index=3}


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


def _fmt2(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _fmt2pct(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def _fmt2comma(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):,.2f}"
    except Exception:
        return "—"


def show(prefill_symbol: str | None = None) -> None:
    st.header("📈 股票")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input(
        "輸入股票代碼或名稱",
        value=default_symbol,
        key="stock_symbol",
        on_change=_sync_symbol_from_input,
    )
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

        # 標題 + 右上角加入觀察
        c1, c2 = st.columns([1, 0.15])
        with c1:
            st.subheader(f"{name or ticker}（{ticker}）")
        with c2:
            if st.button("＋ 加入觀察", key="btn_watch_stock"):
                add_to_watchlist("stock", ticker, name or ticker)

        # ======= Top KPI（**無 Treynor**）=======
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", _fmt2(stats.get("Alpha")))
            st.caption(_tag(stats.get("Alpha"), 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", _fmt2(stats.get("Sharpe Ratio")))
            st.caption(_tag(stats.get("Sharpe Ratio"), 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", _fmt2(stats.get("Beta")))
            st.caption("相對市場波動")

        # ======= 風險摘要（不含 Treynor）=======
        grades = {
            "Alpha": grade_alpha(stats.get("Alpha")),
            "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
        }
        v_de = stats.get("負債權益比")
        v_cr = stats.get("流動比率")
        v_roe = stats.get("ROE")
        grades["負債權益比"] = grade_debt_equity(v_de if pd.notna(v_de) else None)
        grades["流動比率"] = grade_current_ratio(v_cr if pd.notna(v_cr) else None)
        grades["ROE"] = grade_roe(v_roe if pd.notna(v_roe) else None)

        crit, warn, _ = summarize(grades)
        if crit:
            st.warning("⚠ 風險摘要：**" + "、".join(crit) + "** 未達標。")
        elif warn:
            st.info("⚠ 注意：**" + "、".join(warn) + "** 表現普通。")
        else:
            st.success("✅ 指標狀態良好。")

        # ======= 財務列（全部顯示數字）=======
        equity = stats.get("Equity")
        eps_ttm = stats.get("EPS_TTM")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.metric("負債權益比", _fmt2(v_de))
        with col_b:
            st.metric("流動比率", _fmt2(v_cr))
        with col_c:
            st.metric("ROE", _fmt2pct(v_roe))
        with col_d:
            st.metric("股東權益", _fmt2comma(equity))
        with col_e:
            st.metric("EPS (TTM)", _fmt2(eps_ttm))

        # ======= 圖 =======
        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖（日 K）")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")
