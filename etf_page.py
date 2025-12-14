import streamlit as st
from stock_utils import get_metrics, find_ticker_by_name, is_etf
from chart_utils import plot_candlestick_with_ma
import yfinance as yf
from datetime import datetime, timedelta

def show():
    st.header("📊 ETF 專區")

    user_input = st.text_input("輸入 ETF 名稱或代碼", "")
    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    ticker = find_ticker_by_name(user_input.strip().upper())

    # ➤ 若不是 ETF，就不允許在 ETF 區查詢
    if not is_etf(ticker):
        st.error("⚠️ 這不是 ETF，請改至『股票專區』查詢。")
        return

    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    rf = 0.01
    mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

    def tag(val, thr, greater=True):
        if val is None:
            return "❓"
        return "✅" if (val >= thr if greater else val <= thr) else "❗"

    try:
        stats = get_metrics(ticker, mkt, rf, start, end, is_etf=True)
        if stats:
            st.write(f"📊 {stats['name']} ({ticker})")

            st.dataframe({
                "Alpha": [f"{stats['Alpha']} {tag(stats['Alpha'],0)}"],
                "Sharpe Ratio": [f"{stats['Sharpe Ratio']} {tag(stats['Sharpe Ratio'],1)}"],
                "Beta": [stats['Beta']],
                "MADR": [f"{stats['MADR']} {tag(stats['MADR'],0.01, greater=False)}"],
            })

            df = stats["df"]
            fig = plot_candlestick_with_ma(df, title=f"{stats['name']} ({ticker}) 技術圖")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("查無 ETF 資料或資料不足。")
    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")
