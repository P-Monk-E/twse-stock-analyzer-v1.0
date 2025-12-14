# /mnt/data/etf_page.py
import streamlit as st
from stock_utils import get_metrics, find_ticker_by_name
from chart_utils import plot_candlestick_with_ma
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import math

def _sync_symbol_from_input():
    txt = (st.session_state.get("etf_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]

def _tag(val, thr, greater=True):
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    return "✅" if (val >= thr if greater else val <= thr) else "❗"

def show(prefill_symbol: str | None = None):
    st.header("📊 ETF 專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input(
        "輸入 ETF 名稱或代碼",
        value=default_symbol,
        key="etf_symbol",
        on_change=_sync_symbol_from_input,
    )
    user_input = (st.session_state.get("etf_symbol") or "").strip().upper()
    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt, rf, start, end, is_etf=True)
        if not stats:
            st.warning("查無 ETF 資料或資料不足。")
            return

        st.write(f"📊 {stats['name']} ({ticker})")
        st.markdown(f"**Alpha:** {stats['Alpha']} {_tag(stats['Alpha'],0)}")
        st.markdown(f"**Sharpe Ratio:** {stats['Sharpe Ratio']} {_tag(stats['Sharpe Ratio'],1)}")
        st.markdown(f"**Beta:** {stats['Beta']}")
        st.markdown(f"**MADR:** {stats['MADR']}")

        df = stats["df"]
        fig = plot_candlestick_with_ma(df, title=f"{stats['name']} ({ticker}) 技術圖")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
