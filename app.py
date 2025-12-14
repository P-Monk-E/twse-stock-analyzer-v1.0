from __future__ import annotations

import streamlit as st
import stocks_page
import etf_page
import portfolio_page

PAGES = ["股票", "ETF", "庫存"]

def main() -> None:
    st.sidebar.header("主選單")
    nav = st.sidebar.radio("選擇頁面", PAGES, index=0, key="nav_page")

    # why: 返回同頁時預填
    q_symbol = st.query_params.get("symbol")

    if nav == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif nav == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif nav == "庫存":
        portfolio_page.show(prefill_symbol=q_symbol)

if __name__ == "__main__":
    main()
