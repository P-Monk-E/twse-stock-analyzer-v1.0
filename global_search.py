# =========================================
# /mnt/data/global_search.py  （工具：全站搜尋 + 名稱同步）
# =========================================
from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, is_etf, TICKER_NAME_MAP
from names_store import get as name_get, set as name_set

def merge_names_into_builtin_map() -> None:
    """把 names.json 內容合併進 TICKER_NAME_MAP（就地更新 dict，不改原檔）。"""
    try:
        # names_store.get 沒有列舉功能，利用 st.session_state 暫存快取
        cache_key = "_names_cache_map"
        if cache_key not in st.session_state:
            # 讀一次 yfinance 取名會很慢，這裡只合併已知的 names.json（若不存在就跳過）
            # 為了不改 names_store.py，我們用約定的「私有」方法路徑讀它的 json
            import json, os
            from names_store import NAMES_PATH
            data = {}
            if os.path.exists(NAMES_PATH):
                try:
                    with open(NAMES_PATH, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        if isinstance(raw, dict):
                            data = {str(k).upper(): str(v) for k, v in raw.items() if v}
                except Exception:
                    data = {}
            st.session_state[cache_key] = data

        for k, v in st.session_state.get(cache_key, {}).items():
            TICKER_NAME_MAP[k] = v
    except Exception:
        pass

def _fetch_name_from_yf(symbol: str) -> Optional[str]:
    try:
        tk = yf.Ticker(f"{symbol}.TW")
        info = getattr(tk, "fast_info", None) or {}
        # yfinance fast_info 沒有名字時，退回 info/shortName
        nm = None
        try:
            nm = tk.info.get("shortName") or tk.info.get("longName")
        except Exception:
            nm = None
        # 若取不到，用原始代碼回填
        return str(nm).strip() if nm else None
    except Exception:
        return None

def _save_name_if_new(symbol: str, name_hint: Optional[str] = None) -> None:
    """若 names.json 還沒此代碼，嘗試從 yfinance 取名並寫入，並 merge 回內存。"""
    sym = str(symbol).upper()
    if not sym:
        return
    if name_get(sym):
        # 已有，直接 merge 內存
        TICKER_NAME_MAP[sym] = name_get(sym, TICKER_NAME_MAP.get(sym, "")) or TICKER_NAME_MAP.get(sym, "")
        return
    nm = name_hint or _fetch_name_from_yf(sym) or TICKER_NAME_MAP.get(sym, "")
    if nm:
        try:
            name_set(sym, nm)
        except Exception:
            pass
        TICKER_NAME_MAP[sym] = nm  # 合併到內存

def render_global_search() -> Optional[Tuple[str, str, str]]:
    """
    在側邊欄顯示一個全站搜尋框，輸入股票 / ETF 名稱或代碼；
    - 自動判斷股票/ETF
    - 導航到相對應頁籤
    - 同步該頁面的 input value 與 URL query (?symbol=、?nav=)
    - 若缺名稱，會寫入 names.json 並併回 TICKER_NAME_MAP（如 2313）
    回傳 (symbol, name, kind) 或 None
    """
    st.sidebar.markdown("---")
    user_q = st.sidebar.text_input("🔎 全站搜尋（輸入名稱或代碼）", key="global_search_input")

    if not user_q:
        return None

    # 解析為代碼
    symbol = find_ticker_by_name(user_q)
    kind = "ETF" if is_etf(symbol) else "股票"

    # 嘗試補齊名稱（優先 names.json，再用 yfinance 抓一次）
    name = name_get(symbol, TICKER_NAME_MAP.get(symbol, ""))
    if not name:
        _save_name_if_new(symbol)
        name = name_get(symbol, TICKER_NAME_MAP.get(symbol, "")) or symbol

    # 同步 session_state 讓各頁面自帶預填值
    if kind == "ETF":
        st.session_state["etf_symbol"] = symbol
        st.query_params["nav"] = "ETF"
    else:
        st.session_state["stock_symbol"] = symbol
        st.query_params["nav"] = "股票"

    st.query_params["symbol"] = symbol

    # 在側欄即時顯示解析結果
    st.sidebar.caption(f"➡ 導航：{kind}｜{name}（{symbol}）")
    return symbol, name, kind
