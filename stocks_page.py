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
    summarize,
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

def _fmt2(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.2f}"
    except Exception:
        return "—"

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

        # ======= 風險摘要（單一盒子，收斂所有警示）=======
        grades = {}
        grades["Sharpe"] = grade_sharpe(stats.get("Sharpe Ratio"))
        grades["Treynor"] = grade_treynor(stats.get("Treynor"))
        v_de = stats.get("負債權益比");    grades["負債權益比"] = grade_debt_equity(v_de if pd.notna(v_de) else None)
        v_cr = stats.get("流動比率");      grades["流動比率"]   = grade_current_ratio(v_cr if pd.notna(v_cr) else None)
        v_roe = stats.get("ROE");         grades["ROE"]        = grade_roe(v_roe if pd.notna(v_roe) else None)

        crit, warn, good = summarize(grades)
        if crit:
            st.warning("⚠ 風險摘要：**" + "、".join(crit) + "** 指標未達標，請審慎評估。")
        elif warn:
            st.info("⚠ 注意：**" + "、".join(warn) + "** 表現普通。")
        else:
            st.success("✅ 主要指標健康。")

        # ======= 財務比率：一行精簡列示 =======
        def _icon(name: str) -> str:
            return grades[name][0]
        roe_txt = f"{(v_roe*100):.2f}%" if (v_roe is not None and pd.notna(v_roe)) else "—"
        line = (
            f"**負債權益比**：{_fmt2(v_de)} {_icon('負債權益比')} ｜ "
            f"**流動比率**：{_fmt2(v_cr)} {_icon('流動比率')} ｜ "
            f"**ROE**：{roe_txt} {_icon('ROE')}"
        )
        st.markdown(line)

        # ======= 圖表 + 波動提示 =======
        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")
