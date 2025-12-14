import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

PAGES = ["推薦", "股票", "ETF", "庫存"]

# 讀取 URL 參數
query_params = st.experimental_get_query_params()
default_page = query_params.get("page", ["推薦"])[0]

# 防呆：確保 default_page 在 PAGES 裡
if default_page not in PAGES:
    default_page = "推薦"

page = st.sidebar.radio(
    "選擇頁面",
    PAGES,
    index=PAGES.index(default_page)
)

if page == "推薦":
    recommend_page.show()
elif page == "股票":
    stocks_page.show()
elif page == "ETF":
    etf_page.show()
elif page == "庫存":
    portfolio_page.show()
