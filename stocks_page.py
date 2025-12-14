import streamlit as st
from stock_utils import get_metrics, find_ticker_by_name
from chart_utils import plot_candlestick_with_ma
import yfinance as yf
from datetime import datetime, timedelta

def show():
    st.header("📈 股票專區")

    query_symbol = st.experimental_get_query_params().get("symbol", [""])[0]
    default_input = query_symbol if query_symbol else ""
    user_input = st.text_input("輸入股票名稱或代碼", default_input)

    if not user_input:
        st.info("請輸入股票名稱或代碼以查詢。")
        return

    ticker = find_ticker_by_name(user_input.strip().upper())
    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    rf = 0.01
    mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

    try:
        stats = get_metrics(ticker, mkt, rf, start, end)
        if stats:
            st.write(f"📊 {stats['name']} ({ticker})")
            st.dataframe({
                "流動比率": [f"{stats['流動比率']}"],
                "ROE": [f"{stats['ROE']}"],
                "Alpha": [f"{stats['Alpha']}"],
                "Sharpe Ratio": [f"{stats['Sharpe Ratio']}"],
                "Beta": [stats['Beta']],
                "MADR": [f"{stats['MADR']}"],
            })

            df = stats["df"]
            fig = plot_candlestick_with_ma(df, title=f"{stats['name']} ({ticker}) 技術圖")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("查無資料或資料不足。")
    except Exception as e:
        st.error(f"❌ 查詢失敗：{e}")
