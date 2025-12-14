# /mnt/data/app.py  （無變動，附上以便對照/覆蓋）
import streamlit as st

import stocks_page
import etf_page
import portfolio_page

PAGES = ["股票", "ETF", "庫存"]

def _get_current_page() -> str:
    q = st.query_params.get("page", PAGES[0])
    return q if q in PAGES else PAGES[0]

def _on_nav_change():
    st.query_params["page"] = st.session_state["nav_page"]
    if st.session_state["nav_page"] not in ("股票", "ETF") and "symbol" in st.query_params:
        del st.query_params["symbol"]

def main():
    st.set_page_config(page_title="投資儀表板", layout="wide")
    st.sidebar.title("主選單")

    current_page = _get_current_page()

    st.sidebar.radio(
        "選擇頁面",
        PAGES,
        index=PAGES.index(current_page),
        key="nav_page",
        on_change=_on_nav_change,
    )

    q_symbol = st.query_params.get("symbol", "")

    if st.session_state["nav_page"] == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif st.session_state["nav_page"] == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif st.session_state["nav_page"] == "庫存":
        portfolio_page.show()

if __name__ == "__main__":
    main()
