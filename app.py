import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

st.sidebar.title("主選單")
page = st.sidebar.radio("選擇頁面", ["推薦", "股票", "ETF", "庫存"])

if page == "推薦":
    recommend_page.show()
elif page == "股票":
    stocks_page.show()
elif page == "ETF":
    etf_page.show()
elif page == "庫存":
    portfolio_page.show()