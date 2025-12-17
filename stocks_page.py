# =========================================
# /mnt/data/stocks_page.py  （右上角「＋加入觀察」）
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from names_store import get as get_name_override, set as set_name_override
from chart_utils import plot_candlestick_with_ma
from watchlist_page import add_to_watchlist  # 外部API：加入觀察


# --------- helpers ---------
def _fmt2(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"

def _fmt2pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

def _icon(val: Optional[float], good_higher: bool = True) -> str:
    if val is None:
        return "⚪"
    try:
        v = float(val)
    except Exception:
        return "⚪"
    if math.isnan(v) or math.isinf(v):
        return "⚪"
    if good_higher:
        return "🟢" if v > 0 else "🔴" if v < 0 else "⚪"
    else:
        return "🟢" if v < 0 else "🔴" if v > 0 else "⚪"

def _tag(val: Optional[float], target: float, good_higher: bool=True) -> str:
    if val is None:
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    if good_higher:
        return "✅" if v >= target else "❗"
    else:
        return "✅" if v <= target else "❗"


# --------- Page ---------
def show() -> None:
    st.header("📈 股票專區")

    # 搜尋輸入
    q = st.text_input("輸入股票名稱或代碼（例：台積電 或 2330）", key="stock_query")

    if not q:
        st.caption("提示：可輸入中文或代碼（例：台積電、2330）")
        return

    try:
        ticker = find_ticker_by_name(q)
        if not ticker or is_etf(ticker):
            st.warning("請輸入合法的**個股**代碼或名稱。")
            return

        today = datetime.now().date()
        start = today - timedelta(days=365*3)
        end = today
        rf = 0.012  # 假設無風險利率（年化）
        mkt_close = yf.Ticker("^TWII").history(period="3y")["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=False)
        if not stats:
            st.warning("查無資料或資料不足。")
            return

        # 取得名稱並覆寫為自訂名稱（若存在）
        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        name = get_name_override(ticker, name)

        # ---- 標題 + 右上角加入觀察 / 名稱 ----
        c1, c2 = st.columns([1, 0.15])
        with c1:
            st.subheader(f"{name or ticker}（{ticker}）")
        with c2:
            with st.popover("✏️ 名稱", use_container_width=True):
                new_name = st.text_input("自訂名稱（留空則不變）", value=name or ticker, key="stock_custom_name")
                if st.button("儲存名稱", key="btn_save_stock_name"):
                    set_name_override(ticker, new_name or ticker)
                    st.toast("已儲存名稱")
                    name = new_name or ticker
            if st.button("＋ 加入觀察", key="btn_watch_stock"):
                add_to_watchlist("stock", ticker, name or ticker)

        # ======= Top KPI：三欄（無 Treynor）=======
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", _fmt2(stats.get("Alpha")))
            st.caption(_tag(stats.get("Alpha"), 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", _fmt2(stats.get("Sharpe")))
            st.caption(" >1 佳")
        with col3:
            st.metric("Beta", _fmt2(stats.get("Beta")))
            st.caption("相對市場波動")

        # ======= 次要 KPI（股利、ROE、EPS等）=======
        v_eps = stats.get("EPS")
        v_div = stats.get("DividendYield")
        v_roe = stats.get("ROE")
        equity = stats.get("Equity")
        eps_ttm = stats.get("EPS_TTM")

        line = (
            f"**殖利率**：{_fmt2pct(v_div)} {_icon(v_div)} ｜ "
            f"**ROE**：{_fmt2pct(v_roe)} {_icon(v_roe)} ｜ "
            f"**股東權益**：{_fmt2(equity)} ｜ "
            f"**EPS(TTM)**：{_fmt2(eps_ttm)}"
        )
        st.markdown(line)

        # ======= 圖表 =======
        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")
