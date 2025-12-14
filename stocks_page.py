import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from stock_utils import get_metrics, find_ticker_by_name
from chart_utils import plot_candlestick_with_ma

def show():
    st.header("📈 股票專區")

    # 🔁 接收跨頁導入的代碼
    prefill = st.session_state.get("redirect_symbol", "")
    user_input = st.text_input("輸入股票名稱或代碼", value=prefill)
    st.session_state["redirect_symbol"] = ""  # 清空，避免殘留

    # 🔄 使用者修正分類：這是 ETF
    if st.button("🟦 這是 ETF"):
        st.session_state["redirect_symbol"] = user_input
        st.session_state["page"] = "ETF"
        st.rerun()   # ✅ 正確用法（不是 experimental_rerun）

    if not user_input:
        st.info("請輸入股票名稱或代碼以查詢。")
        return

    ticker = find_ticker_by_name(user_input.strip().upper())

    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    rf = 0.01
    mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

    def tag(val, thr, greater=True):
        if val is None:
            return "❓"
        return "✅" if (val >= thr if greater else val <= thr) else "❗"

    try:
        stats = get_metrics(ticker, mkt, rf, start, end)
        if not stats:
            st.warning("查無資料或資料不足。")
            return

        st.subheader(f"{stats['name']} ({ticker})")

        st.dataframe({
            "流動比率": [f"{stats['流動比率']} {tag(stats['流動比率'],1.25)}"],
            "ROE": [f"{stats['ROE']} {tag(stats['ROE'],0.08)}"],
            "Alpha": [f"{stats['Alpha']} {tag(stats['Alpha'],0)}"],
            "Sharpe Ratio": [f"{stats['Sharpe Ratio']} {tag(stats['Sharpe Ratio'],1)}"],
            "Beta": [stats['Beta']],
            "MADR": [f"{stats['MADR']} {tag(stats['MADR'],0.01, greater=False)}"],
        })

        fig = plot_candlestick_with_ma(
            stats["df"],
            title=f"{stats['name']} ({ticker}) 技術圖"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 查詢失敗：{e}")
