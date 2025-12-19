# =========================================
# /mnt/data/etf_page.py
# 修正：find_ticker_by_name 僅回傳代碼 → 只接 1 值；name 由 TICKER_NAME_MAP 取
# 功能：60m/日/週/月 K 線、滑輪縮放、十字線、show() 相容 app.py
# =========================================
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

from risk_grading import grade_alpha, grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning
from stock_utils import find_ticker_by_name, get_metrics, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_ma, resample_ohlc, PLOTLY_TV_CONFIG

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_intraday(ticker: str, interval: str = "60m", period: str = "60d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close"]].dropna(how="any")
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

def _prepare_tf_df(ticker: str, base_daily_df: pd.DataFrame, tf_label: str) -> Tuple[pd.DataFrame, str]:
    if tf_label == "60m":
        df = _download_ohlc_intraday(ticker, "60m", "60d")
        note = "（60 分鐘）"
    elif tf_label == "日":
        df = base_daily_df.copy()
        note = "（日 K）"
    elif tf_label == "週":
        df = resample_ohlc(base_daily_df, "W")
        note = "（週 K）"
    else:
        df = resample_ohlc(base_daily_df, "M")
        note = "（月 K）"
    return df, note

def render(prefill_symbol: Optional[str] = None) -> None:
    st.header("ETF")
    col1, col2 = st.columns([3, 2])
    with col1:
        default_kw = prefill_symbol or st.session_state.get("last_etf_kw", "0050")
        keyword = st.text_input("輸入 ETF 代碼或名稱", value=default_kw)
    with col2:
        tf_label = st.radio("K 線週期", options=["60m", "日", "週", "月"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：0050 或 台灣50）")
        return

    try:
        # 修正重點：find_ticker_by_name 只回傳代碼
        ticker = find_ticker_by_name(keyword)
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_etf_kw"] = keyword

        stats = get_metrics(ticker)
        st.subheader(f"{name or ticker}（{ticker}）")

        sharpe = stats.get("Sharpe Ratio")
        treynor = stats.get("Treynor")
        alpha = stats.get("Alpha")
        st.write(summarize(
            grade_alpha(alpha),
            grade_sharpe(sharpe),
            grade_treynor(treynor),
        ))

        msg = diversification_warning(
            sharpe, treynor,
            non_sys_thr=float(st.session_state.get("non_sys_thr", 0.5)),
            sys_thr=float(st.session_state.get("sys_thr", 0.5)),
        )
        if msg:
            st.warning(msg)

        base_df: pd.DataFrame = stats["df"].copy()
        if not isinstance(base_df.index, pd.DatetimeIndex):
            base_df.index = pd.to_datetime(base_df.index)

        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf_label)
        if tf_df.empty:
            st.error("查無對應週期的價格資料。")
        else:
            title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
            fig = plot_candlestick_with_ma(tf_df, title=title, uirevision_key=f"{ticker}_{tf_label}")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

        right = st.columns([1, 1, 1, 1, 1, 1, 1])[-1]
        with right:
            if st.button("＋加入觀察", use_container_width=True):
                from watchlist_page import add_to_watchlist
                add_to_watchlist(symbol_or_name=ticker, name=name, kind_kw="etf")
                st.success("已加入觀察名單")
    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")

# 與 app.py 相容的入口
def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
