# /mnt/data/etf_page.py
import streamlit as st
from stock_utils import get_metrics, find_ticker_by_name
from chart_utils import plot_candlestick_with_ma
import yfinance as yf
from datetime import datetime, timedelta
import math
import pandas as pd

def show(prefill_symbol=None):
    st.header("📊 ETF 專區")

    query_symbol = st.experimental_get_query_params().get("symbol", [None])[0]
    default_symbol = query_symbol if query_symbol else prefill_symbol or ""
    user_input = st.text_input("輸入 ETF 名稱或代碼", value=default_symbol)

    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    ticker = find_ticker_by_name(user_input.strip().upper())
    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    rf = 0.01
    mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

    def tag(val, thr, greater=True):
        if val is None or (isinstance(val, float) and math.isnan(val)) or (isinstance(val, (int, float)) and pd.isna(val)):
            return "❓"
        return "✅" if (val >= thr if greater else val <= thr) else "❗"

    try:
        # ETF：確保 is_etf=True
        stats = get_metrics(ticker, mkt, rf, start, end, is_etf=True)
        if stats:
            st.write(f"📊 {stats['name']} ({ticker})")

            st.markdown(f"**Alpha:** {stats['Alpha']} {tag(stats['Alpha'],0)}")
            st.markdown(f"**Sharpe Ratio:** {stats['Sharpe Ratio']} {tag(stats['Sharpe Ratio'],1)}")
            st.markdown(f"**Beta:** {stats['Beta']}")
            st.markdown(f"**MADR:** {stats['MADR']}")

            df = stats["df"]
            fig = plot_candlestick_with_ma(df, title=f"{stats['name']} ({ticker}) 技術圖")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("查無 ETF 資料或資料不足。")
    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
