import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

st.sidebar.title("選擇頁面")

# 🔑 統一用 session_state 控制頁面
if "page" not in st.session_state:
    st.session_state["page"] = "推薦"

pages = ["推薦", "股票", "ETF", "庫存"]
current_index = pages.index(st.session_state["page"])

selected_page = st.sidebar.radio(
    "主選單",
    pages,
    index=current_index
)

st.session_state["page"] = selected_page

if selected_page == "推薦":
    recommend_page.show()
elif selected_page == "股票":
    stocks_page.show()
elif selected_page == "ETF":
    etf_page.show()
elif selected_page == "庫存":
    portfolio_page.show()
