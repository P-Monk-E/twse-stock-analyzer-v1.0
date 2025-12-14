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


def _delete_position(idx: int) -> None:
    data = _load_portfolio()
    if 0 <= idx < len(data):
        data.pop(idx)
        _save_portfolio()
        st.success("已刪除。")
        st.rerun()


def _sell_position(idx: int, sell_qty: int, sell_date: date) -> None:
    data = _load_portfolio()
    if not (0 <= idx < len(data)):
        st.warning("找不到該筆持股。"); return
    pos = data[idx]
    cur_qty = int(pos.get("qty", 0))
    if sell_qty <= 0:
        st.warning("賣出數量需大於 0。"); return
    if sell_qty > cur_qty:
        st.warning("賣出數量不可大於目前持股。"); return

    pos["qty"] = cur_qty - sell_qty
    pos.setdefault("sell_logs", []).append({"date": sell_date.isoformat(), "qty": int(sell_qty)})
    if pos["qty"] == 0:
        data.pop(idx)  # why: 完全賣出則移除
        st.info("此筆持股已全部賣出並移除。")
    _save_portfolio()
    st.success("已更新持股。")
    st.rerun()


def _open_confirm(action: Dict[str, Any]) -> None:
    st.session_state["confirm"] = action


def _clear_confirm() -> None:
    st.session_state.pop("confirm", None)


def _show_confirm_ui() -> None:
    if "confirm" not in st.session_state:
        return
    info = st.session_state["confirm"]
    act = info.get("type"); idx = info.get("idx")

    if act == "delete":
        title = "確認刪除"
        msg = f"確定要 **刪除** 第 {idx + 1} 筆持股嗎？此動作無法復原。"
    elif act == "sell":
        title = "確認賣出"
        msg = f"確定要於 **{info.get('sell_date')}** 賣出 **{info.get('sell_qty')} 股**（第 {idx + 1} 筆）嗎？"
    else:
        _clear_confirm(); return

    def _on_confirm():
        if act == "delete":
            _clear_confirm(); _delete_position(idx)
        else:
            _clear_confirm(); _sell_position(idx, int(info["sell_qty"]), info["sell_date"])

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            st.write(msg)
            c1, c2 = st.columns(2)
            if c1.button("確認", type="primary", key="confirm_ok"): _on_confirm()
            if c2.button("取消", key="confirm_cancel"): _clear_confirm(); st.rerun()
        _dlg()
        return

    st.warning(f"**{title}**｜{msg}")
    c1, c2 = st.columns(2)
    if c1.button("確認", type="primary", key="fallback_confirm_ok"): _on_confirm()
    if c2.button("取消", key="fallback_confirm_cancel"): _clear_confirm(); st.rerun()


def show(prefill_symbol: str | None = None) -> None:
    st.header("📦 我的庫存")
    _show_confirm_ui()

    data = _load_portfolio()

    # ---- 新增 ----
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
                data.append({"symbol": sym.strip(), "qty": int(qty), "cost": float(cost), "buy_date": buy_date.isoformat()})
                _save_portfolio(); st.success("已加入。"); st.rerun()

    if not data:
        st.info("目前尚未有持股，請先新增。"); return

    # ---- 顯示表格（無操作鈕）----
    rows = []; total_cost = 0.0; total_value = 0.0
    for row in data:
        sym = row.get("symbol"); qty = float(row.get("qty", 0)); cost = float(row.get("cost", 0.0))
        price = get_latest_price(sym); value = (price or 0.0) * qty
        unreal = (price - cost) * qty if price is not None else None
        rows.append({"買入日": (row.get("buy_date") or "—"), "代碼": sym, "股數": qty, "成本/股": cost,
                     "現價": price if price is not None else "—", "市值": value,
                     "未實現損益": unreal if unreal is not None else "—"})
        total_cost += cost * qty; total_value += value

    df = pd.DataFrame(rows)
    if "買入日" in df.columns:
        try:
            df["_d"] = pd.to_datetime(df["買入日"], errors="coerce")
            df.sort_values(by=["_d", "代碼"], ascending=[True, True], inplace=True)
            df.drop(columns=["_d"], inplace=True)
        except Exception:
            pass

    def _style_unrealized(v):
        if isinstance(v, (int, float)):
            if v > 0: return "color:red;"
            if v < 0: return "color:green;"
        return ""

    try:
        styled = df.style.applymap(_style_unrealized, subset=["未實現損益"])
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

    pnl_unrealized = total_value - total_cost
    st.metric("總市值", f"{total_value:,.0f}")
    st.metric("總未實現損益", f"{pnl_unrealized:,.0f}",
              delta_color=("inverse" if pnl_unrealized < 0 else "normal"))

    # ---- 管理持股（移出列外）----
    with st.expander("管理持股（刪除 / 賣出）", expanded=True):
        # 下拉選單先選一筆
        options = [f"{i+1}. {r.get('symbol')}｜買入日:{r.get('buy_date','—')}｜股數:{r.get('qty')}" for i, r in enumerate(data)]
        sel_idx = st.selectbox("選擇要操作的持股", options=range(len(options)), format_func=lambda i: options[i], key="mgmt_sel")

        cur = data[sel_idx]; cur_qty = int(cur.get("qty", 0))
        st.caption(f"目前選擇：{cur.get('symbol')}｜買入日 {cur.get('buy_date','—')}｜可用股數 {cur_qty}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("刪除這筆持股", key="btn_delete", type="secondary"):
                _open_confirm({"type": "delete", "idx": sel_idx})
        with c2:
            sell_date = st.date_input("賣出日", value=date.today(), key="sell_date_global")
            sell_qty = st.number_input("賣出數量", min_value=1, max_value=max(cur_qty, 1),
                                       value=min(100, max(cur_qty, 1)), step=1, key="sell_qty_global")
            if st.button("賣出", key="btn_sell", type="primary"):
                _open_confirm({"type": "sell", "idx": sel_idx, "sell_qty": int(sell_qty), "sell_date": sell_date})

    _show_confirm_ui()
