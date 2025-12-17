# =========================================
# /mnt/data/app_unified.py  （新增：全站搜尋 + 名稱同步，不改原檔）
# 用法：streamlit run /mnt/data/app_unified.py
# =========================================
from __future__ import annotations

import streamlit as st
import yfinance as yf

# 既有頁面（保持不動）
import stocks_page
import etf_page
import portfolio_page
import watchlist_page

# 既有工具
from stock_utils import find_ticker_by_name, is_etf, TICKER_NAME_MAP
from names_store import get as name_get, set as name_set

from global_search import render_global_search, merge_names_into_builtin_map

PAGES = ["股票", "ETF", "庫存", "觀察名單"]

def _route_to(nav: str, symbol: str | None):
    st.session_state["nav_page"] = nav
    if symbol:
        st.query_params["symbol"] = symbol
    else:
        st.query_params.pop("symbol", None)
    st.query_params["nav"] = nav

def _bootstrap_names():
    # 將 names.json 內容併入 stock_utils.TICKER_NAME_MAP（修正像 2313 不顯示名稱的情況）
    merge_names_into_builtin_map()

def main() -> None:
    st.sidebar.header("主選單")
    _bootstrap_names()

    # ---- 全站搜尋（名稱/代碼，會自動判別股票/ETF 並導航）----
    search_result = render_global_search()
    # 若使用者在搜尋框按下 Enter 或選擇項目，render_global_search 已幫我們處理：
    # - 更新 URL 的 ?symbol= 與 ?nav=
    # - 同步 st.session_state["stock_symbol"] / ["etf_symbol"]
    # - 補齊名稱到 names.json，並 merge 回 TICKER_NAME_MAP

    # ---- 保留原本的分頁選擇 ----
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
