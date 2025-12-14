# /mount/src/twse-stock-analyzer-v1.0/portfolio_page.py
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

SAVE_PATH = "portfolio.json"
REALIZED_PATH = "realized_trades.json"  # 已實現交易紀錄


# ------------------------- Storage -------------------------
def _load_portfolio() -> List[Dict[str, Any]]:
    if "portfolio" in st.session_state and isinstance(st.session_state.portfolio, list):
        return st.session_state.portfolio
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                st.session_state.portfolio = json.load(f)
        except Exception:
            st.session_state.portfolio = []
    else:
        st.session_state.portfolio = []
    return st.session_state.portfolio


def _save_portfolio() -> None:
    try:
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.portfolio, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {SAVE_PATH} 失敗：{e}")


def _load_realized() -> List[Dict[str, Any]]:
    if "realized" in st.session_state and isinstance(st.session_state.realized, list):
        return st.session_state.realized
    if os.path.exists(REALIZED_PATH):
        try:
            with open(REALIZED_PATH, "r", encoding="utf-8") as f:
                st.session_state.realized = json.load(f)
        except Exception:
            st.session_state.realized = []
    else:
        st.session_state.realized = []
    return st.session_state.realized


def _append_realized(rec: Dict[str, Any]) -> None:
    realized = _load_realized()
    realized.append(rec)
    try:
        with open(REALIZED_PATH, "w", encoding="utf-8") as f:
            json.dump(realized, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {REALIZED_PATH} 失敗：{e}")


# ------------------------- Quote -------------------------
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


# ------------------------- Actions -------------------------
def _delete_position(idx: int) -> None:
    data = _load_portfolio()
    if 0 <= idx < len(data):
        data.pop(idx)
        _save_portfolio()
        st.success("已刪除。")
        st.rerun()


def _sell_position(idx: int, sell_qty: int, sell_date: date, sell_price: float) -> None:
    data = _load_portfolio()
    if not (0 <= idx < len(data)):
        st.warning("找不到該筆持股。"); return
    pos = data[idx]
    cur_qty = int(pos.get("qty", 0))
    cost = float(pos.get("cost", 0.0))
    if sell_qty <= 0:
        st.warning("賣出數量需大於 0。"); return
    if sell_qty > cur_qty:
        st.warning("賣出數量不可大於目前持股。"); return
    if sell_price <= 0:
        st.warning("請輸入正確的賣出價格。"); return

    realized_pnl = (sell_price - cost) * sell_qty
    _append_realized(
        {
            "symbol": pos.get("symbol"),
            "sell_date": sell_date.isoformat(),
            "qty": int(sell_qty),
            "sell_price": float(sell_price),
            "buy_cost": cost,
            "pnl": realized_pnl,
        }
    )

    pos["qty"] = cur_qty - sell_qty
    pos.setdefault("sell_logs", []).append(
        {"date": sell_date.isoformat(), "qty": int(sell_qty), "price": float(sell_price)}
    )
    if pos["qty"] == 0:
        data.pop(idx)  # why: 全賣出直接移除
        st.info("此筆持股已全部賣出並移除。")
    _save_portfolio()
    st.success("已更新持股與已實現損益。")
    st.rerun()


# ------------------------- Confirm Dialog -------------------------
def _open_confirm(action: Dict[str, Any]) -> None:
    st.session_state["confirm"] = action


def _clear_confirm() -> None:
    st.session_state.pop("confirm", None)


def _show_confirm_ui() -> None:
    info = st.session_state.get("confirm")
    if not info:
        return

    act = info.get("type"); idx = info.get("idx", -1)
    if act == "delete":
        title = "確認刪除"
        msg = f"確定要 **刪除** 第 {idx + 1} 筆持股嗎？此動作無法復原。"
    elif act == "sell":
        title = "確認賣出"
        msg = (
            f"確定要於 **{info.get('sell_date')}** 以 **{info.get('sell_price'):.4f}**"
            f" 價格賣出 **{info.get('sell_qty')} 股**（第 {idx + 1} 筆）嗎？"
        )
    else:
        _clear_confirm(); return

    def _on_confirm():
        if act == "delete":
            _clear_confirm(); _delete_position(idx)
        else:
            _clear_confirm(); _sell_position(idx, int(info["sell_qty"]), info["sell_date"], float(info["sell_price"]))

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            st.write(msg)
            c1, c2 = st.columns(2)
            if c1.button("確認", type="primary", key="confirm_ok"): _on_confirm()
            if c2.button("取消", key="confirm_cancel"): _clear_confirm(); st.rerun()
        _dlg()
    else:
        st.warning(f"**{title}**｜{msg}")
        c1, c2 = st.columns(2)
        if c1.button("確認", type="primary", key="fallback_ok"): _on_confirm()
        if c2.button("取消", key="fallback_cancel"): _clear_confirm(); st.rerun()


# ------------------------- Page -------------------------
def show(prefill_symbol: str | None = None) -> None:
    st.header("📦 我的庫存")
    _show_confirm_ui()

    data = _load_portfolio()
    realized = _load_realized()

    # ---- 新增持股 ----
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
                data.append(
                    {"symbol": sym.strip(), "qty": int(qty), "cost": float(cost), "buy_date": buy_date.isoformat()}
                )
                _save_portfolio(); st.success("已加入。"); st.rerun()

    # ---- 已實現損益（即便無持倉也顯示）----
    total_realized = sum(float(x.get("pnl", 0.0)) for x in realized)

    if not data:
        st.info("目前尚未有持股，請先新增。")
        st.metric("已實現損益", f"{total_realized:,.4f}")
        return

    # ---- 表格資料（保持數值，格式化階段再加千分位/百分比）----
    rows = []; principal = 0.0; total_value = 0.0
    for row in data:
        sym = row.get("symbol")
        qty = float(row.get("qty", 0.0))
        cost = float(row.get("cost", 0.0))
        price = get_latest_price(sym)
        value = (price or 0.0) * qty
        unreal = (price - cost) * qty if price is not None else float("nan")
        rate_pct = ((price - cost) / cost * 100.0) if (price is not None and cost > 0) else float("nan")

        rows.append(
            {
                "買入日": (row.get("buy_date") or "—"),
                "代碼": sym,
                "股數": qty,
                "成本/股": cost,
                "現價": price,
                "市值": value,
                "未實現損益": unreal,
                "回報率%": rate_pct,
            }
        )
        principal += cost * qty
        total_value += value

    df = pd.DataFrame(rows)
    try:
        df["_d"] = pd.to_datetime(df["買入日"], errors="coerce")
        df.sort_values(by=["_d", "代碼"], ascending=[True, True], inplace=True)
        df.drop(columns=["_d"], inplace=True)
    except Exception:
        pass

    # 色彩：正紅、負綠
    def _style_num(v):
        if isinstance(v, (int, float)) and pd.notna(v):
            if v > 0: return "color:red;"
            if v < 0: return "color:green;"
        return ""

    # 指定格式（金額/數量4位+千分位，百分比2位）
    try:
        styled = (
            df.style
            .format(
                {
                    "股數": "{:,.4f}",
                    "成本/股": "{:,.4f}",
                    "現價": "{:,.4f}",
                    "市值": "{:,.4f}",
                    "未實現損益": "{:,.4f}",
                    "回報率%": "{:.2f}%",  # 僅顯示百分比符號與兩位小數
                },
                na_rep="—",
            )
            .applymap(_style_num, subset=["未實現損益"])
            .applymap(_style_num, subset=["回報率%"])
        )
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

    # ---- 總計（4位 / 2位%）----
    pnl_unrealized = total_value - principal
    total_return_rate = (pnl_unrealized / principal * 100.0) if principal > 0 else 0.0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("總市值", f"{total_value:,.4f}")
        st.caption(f"本金：{principal:,.4f}")
    with c2:
        st.metric(
            "總未實現損益",
            f"{pnl_unrealized:,.4f}",
            delta=f"{total_return_rate:.2f}%",
            delta_color=("inverse" if pnl_unrealized < 0 else "normal"),
        )
        st.caption(f"已實現損益：{total_realized:,.4f}")

    # ---- 管理持股（刪除 / 賣出）----
    with st.expander("管理持股（刪除 / 賣出）", expanded=True):
        options = [f"{i+1}. {r.get('symbol')}｜買入日:{r.get('buy_date','—')}｜股數:{r.get('qty')}" for i, r in enumerate(data)]
        sel_idx = st.selectbox("選擇要操作的持股", options=range(len(options)), format_func=lambda i: options[i], key="mgmt_sel")

        cur = data[sel_idx]; cur_qty = int(cur.get("qty", 0))
        st.caption(f"目前選擇：{cur.get('symbol')}｜買入日 {cur.get('buy_date','—')}｜可用股數 {cur_qty:,}")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("刪除這筆持股", key="btn_delete", type="secondary"):
                _open_confirm({"type": "delete", "idx": sel_idx})
        with c2:
            sell_date = st.date_input("賣出日", value=date.today(), key="sell_date_global")
            sell_qty = st.number_input(
                "賣出數量", min_value=1, max_value=max(cur_qty, 1),
                value=min(100, max(cur_qty, 1)), step=1, key="sell_qty_global"
            )
        with c3:
            sell_price = st.number_input("賣出價格", min_value=0.0, value=0.0, step=0.0001, key="sell_price_global")
            if st.button("賣出", key="btn_sell", type="primary"):
                _open_confirm(
                    {
                        "type": "sell", "idx": sel_idx,
                        "sell_qty": int(sell_qty), "sell_date": sell_date,
                        "sell_price": float(sell_price),
                    }
                )

    _show_confirm_ui()
