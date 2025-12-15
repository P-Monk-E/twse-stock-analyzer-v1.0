# =========================================
# /mnt/data/app.py  （加入「觀察名單」頁）
# =========================================
import streamlit as st
import stocks_page
import etf_page
import portfolio_page
import watchlist_page  # 新增

PAGES = ["股票", "ETF", "庫存", "觀察名單"]  # 新增頁籤
# 以上改動以你現有檔案為基礎擴充。 :contentReference[oaicite:0]{index=0}

def main() -> None:
    st.sidebar.header("主選單")
    nav_param = st.query_params.get("nav")
    default_index = PAGES.index(nav_param) if nav_param in PAGES else 0

    nav = st.sidebar.radio("選擇頁面", PAGES, index=default_index, key="nav_page")
    q_symbol = st.query_params.get("symbol")

    if nav == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif nav == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif nav == "庫存":
        portfolio_page.show(prefill_symbol=q_symbol)
    else:
        watchlist_page.show()

if __name__ == "__main__":
    main()
