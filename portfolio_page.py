# /mnt/data/portfolio_page.py
import json
import os
from datetime import date
from typing import Any, Dict, List

import streamlit as st
import yfinance as yf

SAVE_PATH = "portfolio.json"

# --------------------------
# Helpers: IO & Price
# --------------------------
def _load_portfolio() -> List[Dict[str, Any]]:
    if "portfolio" in st.session_state and isinstance(st.session_state.portfolio, list):
        return st.session_state.portfolio
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                st.session_state.portfolio = json.load(f)
                return st.session_state.portfolio
        except Exception:
            pass
    st.session_state.portfolio = []
    return st.session_state.portfolio

def _save_portfolio() -> None:
    try:
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.portfolio, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {SAVE_PATH} 失敗：{e}")

@st.cache_data(ttl=3600)
def get_latest_price(symbol: str):
    s = symbol.upper().strip()
    cands = [s] if s.endswith((".TW", ".TWO")) else [f"{s}.TW", f"{s}.TWO"]
    for c in cands:
        try:
            info = yf.Ticker(c).fast_info
            p = info.get("lastPrice")
            if p:
                return float(p)
        except Exception:
            continue
    return None

# --------------------------
# Query params sync (URL <-> UI)
# --------------------------
def _qp_get(name: str, default: str) -> str:
    return st.query_params.get(name, default)

def _qp_set_or_del(name: str, value: str | None):
    if value is None or value == "":
        if name in st.query_params:
            del st.query_params[name]
    else:
        st.query_params[name] = str(value)

def _sync_search():
    _qp_set_or_del("pf_q", (st.session_state.get("pf_q") or "").strip())

def _sync_min_ret():
    v = st.session_state.get("pf_min_ret")
    _qp_set_or_del("pf_min_ret", None if v is None else str(v))

def _sync_sort():
    _qp_set_or_del("pf_sort", st.session_state.get("pf_sort"))
    _qp_set_or_del("pf_asc", "1" if st.session_state.get("pf_asc") else "0")

# --------------------------
# Page
# --------------------------
def show():
    st.header("📦 庫存")

    portfolio = _load_portfolio()

    # ---- Filters & Sorting (Sidebar) ----
    with st.sidebar.expander("篩選 / 排序", expanded=True):
        default_q = _qp_get("pf_q", "")
        default_min_ret_str = _qp_get("pf_min_ret", "")
        try:
            default_min_ret = float(default_min_ret_str) if default_min_ret_str != "" else 0.0
        except Exception:
            default_min_ret = 0.0
        default_sort = _qp_get("pf_sort", "報酬率")
        default_asc = _qp_get("pf_asc", "0")
        default_asc_bool = default_asc == "1"

        st.text_input("搜尋代碼（包含字串）", value=default_q, key="pf_q", on_change=_sync_search)
        st.number_input("最小報酬率（%）", value=float(default_min_ret), step=1.0, key="pf_min_ret", on_change=_sync_min_ret)

        sort_fields = ["報酬率", "市值", "成本", "持股", "代碼"]
        if default_sort not in sort_fields:
            default_sort = "報酬率"
        st.selectbox("排序欄位", sort_fields, index=sort_fields.index(default_sort), key="pf_sort", on_change=_sync_sort)
        st.checkbox("升冪", value=default_asc_bool, key="pf_asc", on_change=_sync_sort)

    # ---- Add position ----
    with st.form("add_form", clear_on_submit=False):
        st.subheader("新增持股")
        c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 1.6])
        with c1:
            code = st.text_input("代碼", key="add_code")
        with c2:
            shares = st.number_input("股數", min_value=0, step=1, key="add_shares")
        with c3:
            cost = st.number_input("成本/股", min_value=0.0, step=0.1, key="add_cost")
        with c4:
            buy_date = st.date_input("買進日", value=date.today(), key="add_date")

        submitted = st.form_submit_button("➕ 新增")
        if submitted:
            if not code:
                st.warning("請輸入代碼")
            else:
                portfolio.append({
                    "ticker": code.strip().upper(),
                    "shares": int(shares),
                    "cost": float(cost),
                    "date": str(buy_date),
                })
                _save_portfolio()
                st.success("已新增")
                st.experimental_rerun()

    st.divider()

    # ---- Compute values ----
    rows = []
    total_capital = 0.0
    total_value = 0.0
    total_unrealized = 0.0
    total_realized = 0.0

    for idx, pos in enumerate(portfolio):
        t = pos.get("ticker", "").upper()
        sh = float(pos.get("shares", 0))
        cost = float(pos.get("cost", 0.0))
        latest = get_latest_price(t)
        price = 0.0 if latest is None else latest

        value = price * sh
        profit = (price - cost) * sh
        ret = ((price - cost) / cost * 100) if cost > 0 else 0.0

        rows.append({
            "idx": idx,
            "代碼": t,
            "持股": sh,
            "成本": cost,
            "現價": price,
            "市值": value,
            "損益": profit,
            "報酬率": ret,
            "date": pos.get("date", ""),
        })

        total_capital += cost * sh
        total_value += value
        total_unrealized += profit

    # ---- Apply filter ----
    q = (st.session_state.get("pf_q") or "").strip().upper()
    min_ret = float(st.session_state.get("pf_min_ret") or 0.0)
    if q:
        rows = [r for r in rows if q in r["代碼"]]
    rows = [r for r in rows if r["報酬率"] >= min_ret]

    # ---- Sorting ----
    sort_key = st.session_state.get("pf_sort") or "報酬率"
    asc = bool(st.session_state.get("pf_asc"))
    def _keyfn(r):
        if sort_key == "代碼":
            return r["代碼"]
        return float(r.get(sort_key, 0.0))
    rows.sort(key=_keyfn, reverse=not asc)

    # ---- Render table ----
    if not rows:
        st.info("沒有符合條件的持股。")
    else:
        for r in rows:
            i = r["idx"]
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.6])
            with col1:
                st.markdown(f"**{r['代碼']}**")
                st.caption(r["date"])
            with col2:
                st.metric("持股", int(r["持股"]))
            with col3:
                st.metric("成本/股", round(r["成本"], 2))
            with col4:
                st.metric("現價", round(r["現價"], 2))
            with col5:
                st.metric("市值", round(r["市值"], 2))
            with col6:
                st.metric("報酬率(%)", round(r["報酬率"], 2))
            with col7:
                if st.button("🗑️", key=f"del_{i}"):
                    try:
                        st.session_state.portfolio.pop(i)
                        _save_portfolio()
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"刪除失敗：{e}")

    # ---- Totals ----
    total_return = ((total_value - total_capital) / total_capital * 100) if total_capital > 0 else 0.0
    st.markdown(f"🔥 **總市值：{round(total_value,2)}**")
    st.markdown(f"💵 **總投入資金：{round(total_capital,2)}**")
    st.markdown(f"📉 **總報酬率：{round(total_return,2)}%**")
    st.caption(f"未實現損益：{round(total_unrealized,2)} 元")
    st.caption(f"🟩 已實現損益：{round(total_realized,2)} 元")
