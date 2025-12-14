import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

# ---------------------------
# 讀取 URL query 參數
# ---------------------------
query_params = st.experimental_get_query_params()
q_page = query_params.get("page", [None])[0]
q_symbol = query_params.get("symbol", [None])[0]

st.sidebar.title("主選單")

# 先依照 query 參數決定哪一頁
pages = ["推薦", "股票", "ETF", "庫存"]
if q_page in pages:
    current_page = q_page
else:
    # 沒有 query 參數時用側欄選單
    current_page = st.sidebar.radio("選擇頁面", pages)

# 顯示對應頁面
if current_page == "推薦":
    recommend_page.show()
elif current_page == "股票":
    stocks_page.show(q_symbol)
elif current_page == "ETF":
    etf_page.show(q_symbol)
elif current_page == "庫存":
    portfolio_page.show()
