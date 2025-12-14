import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

query_params = st.experimental_get_query_params()
default_page = query_params.get("page", ["推薦"])[0]
page = st.sidebar.radio("選擇頁面", ["推薦", "股票", "ETF", "庫存"],
                        index=["推薦", "股票", "ETF", "庫存"].index(default_page))

if page == "推薦":
    recommend_page.show()
elif page == "股票":
    stocks_page.show()
elif page == "ETF":
    etf_page.show()
elif page == "庫存":
    portfolio_page.show()
