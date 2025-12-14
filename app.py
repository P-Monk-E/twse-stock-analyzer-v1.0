# /mnt/data/app.py
import streamlit as st

import recommend_page
import stocks_page
import etf_page
import portfolio_page

PAGES = ["推薦", "股票", "ETF", "庫存"]

def _get_current_page() -> str:
    q = st.query_params.get("page", PAGES[0])
    return q if q in PAGES else PAGES[0]

def _on_nav_change():
    # 為避免帶著舊 symbol 跳到非 股票/ETF 頁面
    st.query_params["page"] = st.session_state["nav_page"]
    if st.session_state["nav_page"] not in ("股票", "ETF") and "symbol" in st.query_params:
        del st.query_params["symbol"]

def main():
    st.set_page_config(page_title="投資儀表板", layout="wide")

    st.sidebar.title("主選單")

    # 目前頁面（由 URL 或預設）
    current_page = _get_current_page()

    # 側欄選單（與 URL 同步）
    st.sidebar.radio(
        "選擇頁面",
        PAGES,
        index=PAGES.index(current_page),
        key="nav_page",
        on_change=_on_nav_change,
    )

    # 取得 URL 的 symbol（僅股票/ETF會用到）
    q_symbol = st.query_params.get("symbol", "")

    # 路由
    if st.session_state["nav_page"] == "推薦":
        recommend_page.show()
    elif st.session_state["nav_page"] == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif st.session_state["nav_page"] == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif st.session_state["nav_page"] == "庫存":
        portfolio_page.show()

if __name__ == "__main__":
    main()
