import streamlit as st
import yfinance as yf
from stock_utils import get_metrics, TICKER_NAME_MAP
from alert_rules import get_recommendations
from datetime import datetime, timedelta

def show():
    st.header("🔥 推薦")

    all_stock = list(TICKER_NAME_MAP.keys())
    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    rf = 0.01
    mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

    st.subheader("📈 推薦股票")
    recs = get_recommendations(all_stock, mkt, rf, start, end)
    if recs:
        for t, data in recs:
            st.write(f"{data['name']} ({t}) — Alpha {data['Alpha']:.2f}, Sharpe {data['Sharpe Ratio']:.2f}")
    else:
        st.write("無推薦股票。")

    st.subheader("📊 推薦ETF")
    rec_etf = get_recommendations(all_stock, mkt, rf, start, end, is_etf=True)
    if rec_etf:
        for t, data in rec_etf:
            st.write(f"{data['name']} ({t}) — Alpha {data['Alpha']:.2f}, Sharpe {data['Sharpe Ratio']:.2f}")
    else:
        st.write("無推薦 ETF。")