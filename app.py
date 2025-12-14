import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

st.sidebar.title("主選單")

# ✅ 修改三：確保頁面能切換與重新導向
if "page" not in st.session_state:
    st.session_state["page"] = "推薦"

selected_page = st.sidebar.radio("選擇頁面", ["推薦", "股票", "ETF", "庫存"], index=["推薦", "股票", "ETF", "庫存"].index(st.session_state["page"]))
st.session_state["page"] = selected_page

if st.session_state["page"] == "推薦":
    recommend_page.show()
elif st.session_state["page"] == "股票":
    stocks_page.show()
elif st.session_state["page"] == "ETF":
    etf_page.show()
elif st.session_state["page"] == "庫存":
    portfolio_page.show()
