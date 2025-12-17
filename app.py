# =========================================
# /mnt/data/app.py  （健壯化呼叫 show()，自動相容不同參數）
# =========================================
from __future__ import annotations

import inspect
import streamlit as st

import stocks_page
import etf_page
import portfolio_page
import watchlist_page

PAGES = ["股票", "ETF", "庫存", "觀察名單"]


def _call_show(mod, q_symbol):
    """健壯呼叫：自動偵測 show() 的參數型態並相容呼叫。
    為了相容舊版頁面（不接受 prefill_symbol）而設計。"""
    try:
        sig = inspect.signature(mod.show)
        params = sig.parameters
        if "prefill_symbol" in params:
            return mod.show(prefill_symbol=q_symbol)
        # 找第一個可以吃位置參數的入口
        accepts_positional = any(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params.values()
        )
        if accepts_positional and len(params) >= 1:
            return mod.show(q_symbol)
        # 無參數版本
        return mod.show()
    except TypeError:
        # 退階重試：位置參數 → 無參數
        try:
            return mod.show(q_symbol)
        except TypeError:
            return mod.show()


def main() -> None:
    st.sidebar.header("主選單")

    # 支援透過 URL 切頁與帶入代碼：./?nav=股票&symbol=2330
    nav_param = st.query_params.get("nav")
    default_index = PAGES.index(nav_param) if nav_param in PAGES else 0
    nav = st.sidebar.radio("選擇頁面", PAGES, index=default_index, key="nav_page")
    q_symbol = st.query_params.get("symbol")

    if nav == "股票":
        _call_show(stocks_page, q_symbol)
    elif nav == "ETF":
        _call_show(etf_page, q_symbol)
    elif nav == "庫存":
        _call_show(portfolio_page, q_symbol)
    else:
        watchlist_page.show()  # 觀察名單頁本身無需帶參數


if __name__ == "__main__":
    main()
