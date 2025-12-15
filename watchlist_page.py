# =========================================
# /mnt/data/watchlist_page.py  （新增檔：觀察名單 + 共用工具）
# =========================================
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

WATCHLIST_PATH = "watchlist.json"

# --------- storage ---------
def _empty() -> Dict[str, List[Dict]]:
    return {"stocks": [], "etfs": []}

def load_watchlist() -> Dict[str, List[Dict]]:
    if "watchlist" in st.session_state and isinstance(st.session_state.watchlist, dict):
        return st.session_state.watchlist
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                st.session_state.watchlist = json.load(f)
        except Exception:
            st.session_state.watchlist = _empty()
    else:
        st.session_state.watchlist = _empty()
    # 型別守衛
    for k in ("stocks", "etfs"):
        if k not in st.session_state.watchlist or not isinstance(st.session_state.watchlist[k], list):
            st.session_state.watchlist[k] = []
    return st.session_state.watchlist

def save_watchlist() -> None:
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.watchlist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {WATCHLIST_PATH} 失敗：{e}")

def add_to_watchlist(kind: str, symbol: str, name: str) -> None:
    wl = load_watchlist()
    key = "etfs" if kind == "etf" else "stocks"
    symbol_u = symbol.strip().upper()
    # 去重：同代碼只保留一筆
    if any(symbol_u == x.get("symbol", "").upper() for x in wl[key]):
        st.info("已在觀察名單中。")
        return
    wl[key].append({"symbol": symbol_u, "name": name or symbol_u, "added_at": datetime.now().isoformat(timespec="seconds")})
    save_watchlist()
    st.success("已加入觀察名單。")

def remove_from_watchlist(kind: str, symbols: List[str]) -> None:
    wl = load_watchlist()
    key = "etfs" if kind == "etf" else "stocks"
    to_del = {s.upper() for s in symbols}
    wl[key] = [row for row in wl[key] if row.get("symbol", "").upper() not in to_del]
    save_watchlist()
    st.success("已刪除所選項目。")

# --------- page ---------
def show() -> None:
    st.header("👀 觀察名單")
    wl = load_watchlist()

    # 待觀察股票
    with st.expander("待觀察股票", expanded=True):
        df_s = pd.DataFrame(wl["stocks"]) if wl["stocks"] else pd.DataFrame(columns=["symbol", "name", "added_at"])
        st.dataframe(df_s.rename(columns={"symbol": "代碼", "name": "名稱", "added_at": "加入時間"}), use_container_width=True, hide_index=True)
        sel_s = st.multiselect("選擇要刪除的股票", options=[r["symbol"] for r in wl["stocks"]], key="wl_sel_stocks")
        if st.button("刪除選取（股票）"):
            remove_from_watchlist("stock", sel_s)

    # 待觀察ETF
    with st.expander("待觀察ETF", expanded=True):
        df_e = pd.DataFrame(wl["etfs"]) if wl["etfs"] else pd.DataFrame(columns=["symbol", "name", "added_at"])
        st.dataframe(df_e.rename(columns={"symbol": "代碼", "name": "名稱", "added_at": "加入時間"}), use_container_width=True, hide_index=True)
        sel_e = st.multiselect("選擇要刪除的ETF", options=[r["symbol"] for r in wl["etfs"]], key="wl_sel_etfs")
        if st.button("刪除選取（ETF）"):
            remove_from_watchlist("etf", sel_e)
