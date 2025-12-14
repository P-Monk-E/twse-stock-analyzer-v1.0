# =========================================
# portfolio_page.py
# =========================================
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

SAVE_PATH = "portfolio.json"


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
def get_latest_price(symbol: str) -> Optional[float]:
    s = symbol.upper().strip()
    cands = [s] if s.endswith((".TW", ".TWO")) else [f"{s}.TW", f"{s}.TWO"]
    for c in cands:
        try:
            p = yf.Ticker(c).fast_info.get("lastPrice")
            if p:
                return float(p)
        except Exception:
            pass
        try:
            hist = yf.Ticker(c).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return None


def show(prefill_symbol: str | None = None) -> None:
    st.header("📦 我的庫存")

    data = _load_portfolio()

    with st.expander("新增持股", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            sym = st.text_input("代碼（例：2330 或 2330.TW）", value=prefill_symbol or "", key="pf_add_sym")
        with c2:
            qty = st.number_input("股數", min_value=1, value=100, step=100, key="pf_add_qty")
        with c3:
            cost = st.number_input("成本/股", min_value=0.0, value=100.0, step=0.1, key="pf_add_cost")
        with c4:
            buy_date: date = st.date_input("買入日", value=date.today(), key="pf_add_date")
        if st.button("加入", type="primary"):
            if not sym.strip():
                st.warning("請輸入代碼。")
            else:
                # why: 存 ISO-8601 便於排序/跨平台
                data.append(
                    {
                        "symbol": sym.strip(),
                        "qty": int(qty),
                        "cost": float(cost),
                        "buy_date": buy_date.isoformat(),
                    }
                )
                _save_portfolio()
                st.success("已加入。")
                st.rerun()

    if not data:
        st.info("目前尚未有持股，請先新增。")
        return

    rows = []
    total_cost = 0.0
    total_value = 0.0
    for row in data:
        sym = row.get("symbol")
        qty = float(row.get("qty", 0))
        cost = float(row.get("cost", 0.0))
        price = get_latest_price(sym)
        value = (price or 0.0) * qty
        rows.append(
            {
                "買入日": (row.get("buy_date") or "—"),
                "代碼": sym,
                "股數": qty,
                "成本/股": cost,
                "現價": price if price is not None else "—",
                "市值": value,
                "損益": value - cost * qty,
            }
        )
        total_cost += cost * qty
        total_value += value

    df = pd.DataFrame(rows)
    # 按日期排序（有日期的在前），顯示更直覺
    if "買入日" in df.columns:
        try:
            df["_d"] = pd.to_datetime(df["買入日"], errors="coerce")
            df.sort_values(by=["_d", "代碼"], ascending=[True, True], inplace=True)
            df.drop(columns=["_d"], inplace=True)
        except Exception:
            pass

    st.dataframe(df, use_container_width=True)
    pnl = total_value - total_cost
    st.metric("總市值", f"{total_value:,.0f}")
    st.metric("總損益", f"{pnl:,.0f}")
