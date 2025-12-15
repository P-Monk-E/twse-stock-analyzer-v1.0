from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from portfolio_utils import estimate_portfolio_risk, set_portfolio_risk_warning
from portfolio_risk_utils import diversification_warning

SAVE_PATH = "portfolio.json"
REALIZED_PATH = "realized_trades.json"  # 已實現交易紀錄

# ---------- Utils ----------
def guess_is_etf(symbol: str) -> bool:
    return symbol.strip().upper().startswith("00")  # 台灣 ETF 多為 00xxx（簡易判斷）

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

def _fmt4(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.4f}"
    except Exception:
        return "—"

# ----- 風險偵測入口（頁首顯示） -----
warn = st.session_state.get("portfolio_risk_warning")
if warn:
    st.warning(warn)

# ---------- Storage ----------
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

# ---------- Actions ----------
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
        st.warning("找不到該筆持股。")
        return
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
        data.pop(idx)
        st.info("此筆持股已全部賣出並移除。")
    _save_portfolio()
    st.success("已更新持股與已實現損益。")
    st.rerun()

def _fifo_sell(symbol: str, sell_qty: int, sell_date: date, sell_price: float) -> None:
    data = _load_portfolio()
    lots = [(i, r) for i, r in enumerate(data) if str(r.get("symbol")).strip().upper() == symbol.strip().upper()]
    if not lots:
        st.warning("找不到該代碼的持股。"); return
    if sell_qty <= 0:
        st.warning("賣出數量需大於 0。"); return
    if sell_price <= 0:
        st.warning("請輸入正確的賣出價格。"); return

    def _key(t):
        d = t[1].get("buy_date") or ""
        try:
            return (pd.to_datetime(d), t[0])
        except Exception:
            return (pd.Timestamp.max, t[0])

    lots.sort(key=_key)

    remaining = sell_qty
    for idx, lot in lots:
        if remaining <= 0:
            break
        lot_qty = int(lot.get("qty", 0))
        if lot_qty <= 0:
            continue
        take = min(remaining, lot_qty)
        cost = float(lot.get("cost", 0.0))
        pnl = (sell_price - cost) * take
        _append_realized(
            {
                "symbol": lot.get("symbol"),
                "sell_date": sell_date.isoformat(),
                "qty": int(take),
                "sell_price": float(sell_price),
                "buy_cost": cost,
                "pnl": pnl,
                "buy_date": lot.get("buy_date", None),
            }
        )
        lot["qty"] = lot_qty - take
        lot.setdefault("sell_logs", []).append(
            {"date": sell_date.isoformat(), "qty": int(take), "price": float(sell_price), "mode": "FIFO"}
        )
        remaining -= take

    st.session_state.portfolio = [r for r in data if int(r.get("qty", 0)) > 0]
    _save_portfolio()

    sold = sell_qty - max(remaining, 0)
    if sold <= 0:
        st.warning("沒有可賣出的數量。"); return
    if remaining > 0:
        st.info(f"持股不足，已依 FIFO 賣出 {sold} 股。")
    else:
        st.success(f"已依 FIFO 完成賣出 {sold} 股。")
    st.rerun()

# ---------- Confirm Dialog ----------
def _render_confirm_dialog() -> None:
    info = st.session_state.get("confirm")
    if not info:
        return

    act = info.get("type")
    with st.container():
        st.warning("請再次確認以下操作無誤：")
        if act == "delete":
            idx = info.get("idx")
            data = _load_portfolio()
            if 0 <= idx < len(data):
                row = data[idx]
                st.write(f"將 **刪除**：{row.get('symbol')}｜買入日 {row.get('buy_date','—')}｜股數 {row.get('qty')}")
        elif act == "sell":
            st.write(
                f"將 **賣出**：索引 {info.get('idx')}｜數量 {info.get('sell_qty')}｜"
                f"價格 {info.get('sell_price')}｜日期 {info.get('sell_date')}"
            )
        elif act == "sell_fifo":
            st.write(
                f"將 **FIFO 賣出**：代碼 {info.get('symbol')}｜數量 {info.get('sell_qty')}｜"
                f"價格 {info.get('sell_price')}｜日期 {info.get('sell_date')}"
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 確認執行", key="btn_confirm_yes"):
                try:
                    if act == "delete":
                        _delete_position(int(info.get("idx", -1)))
                    elif act == "sell":
                        _sell_position(int(info.get("idx", -1)),
                                       int(info.get("sell_qty", 0)),
                                       info.get("sell_date"),
                                       float(info.get("sell_price", 0.0)))
                    elif act == "sell_fifo":
                        _fifo_sell(str(info.get("symbol")),
                                   int(info.get("sell_qty", 0)),
                                   info.get("sell_date"),
                                   float(info.get("sell_price", 0.0)))
                finally:
                    st.session_state.pop("confirm", None)
                    st.rerun()
        with c2:
            if st.button("取消", key="btn_confirm_cancel"):
                st.session_state.pop("confirm", None)
                st.info("已取消操作。")
                st.rerun()

# ---------- Page ----------
def show(prefill_symbol: Optional[str] = None) -> None:
    st.header("📦 我的庫存")

    data = _load_portfolio()
    realized = _load_realized()

    # 風險偵測（近一年估算）
    with st.expander("風險偵測（近一年估算）", expanded=False):
        st.caption("說明：依最近收盤價權重合成組合報酬，市場以 ^TWII（取不到時 ^TAIEX，最後 ^GSPC），rf=1%。")

        ns_default = float(st.session_state.get("non_sys_thr", 0.5))
        s_default = float(st.session_state.get("sys_thr", 0.5))
        c1, c2 = st.columns(2)
        with c1:
            non_sys_thr = st.slider("非系統性門檻：Treynor − Sharpe >", 0.1, 2.0, ns_default, 0.1)
        with c2:
            sys_thr = st.slider("系統性門檻：Treynor − Sharpe < −", 0.1, 2.0, s_default, 0.1)
        st.session_state["non_sys_thr"] = float(non_sys_thr)
        st.session_state["sys_thr"] = float(sys_thr)

        if st.button("估算並產生風險警告", type="primary"):
            sharpe, treynor, dbg = estimate_portfolio_risk(data)
            diff = None if (sharpe is None or treynor is None) else (treynor - sharpe)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sharpe", _fmt4(sharpe))
            col_b.metric("Treynor", _fmt4(treynor))
            col_c.metric("Diff (T−S)", _fmt4(diff))

            if sharpe is None and treynor is None:
                st.warning(f"⚠ 無法估算：{dbg}")
            elif treynor is None:
                st.warning(f"⚠ 僅估出 Sharpe，Treynor 無法估算：{dbg}")
            else:
                set_portfolio_risk_warning(sharpe, treynor, non_sys_thr=non_sys_thr, sys_thr=sys_thr)
                msg = diversification_warning(sharpe, treynor, non_sys_thr=non_sys_thr, sys_thr=sys_thr)
                if msg:
                    st.warning(msg)
                else:
                    st.success("✅ 估算完成，未偵測到明顯分散/系統性風險失衡。")

    # ======== 持股明細 ========
    def fmt4(x: Optional[float]) -> str:
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            return f"{float(x):,.4f}"
        except Exception:
            return "—"

    def fmtpct2(x: Optional[float]) -> str:
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            return f"{float(x):.2f}%"
        except Exception:
            return "—"

    with st.expander("新增持股", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            sym = st.text_input("代碼（例：2330 或 2330.TW）", value=prefill_symbol or "", key="pf_add_sym")
        with c2:
            qty = st.number_input("股數", min_value=1, value=100, step=100, key="pf_add_qty")
        with c3:
            cost = st.number_input("成本/股", min_value=0.0, value=100.0, step=0.1, key="pf_add_cost")
        with c4:
            buy_date = st.date_input("買入日", value=date.today(), key="pf_add_date")
        if st.button("加入", type="primary"):
            if not sym.strip():
                st.warning("請輸入代碼。")
            else:
                data.append({"symbol": sym.strip(), "qty": int(qty), "cost": float(cost), "buy_date": buy_date.isoformat()})
                _save_portfolio(); st.success("已加入。"); st.rerun()

    total_realized = sum(float(x.get("pnl", 0.0)) for x in realized)

    if not data:
        st.info("目前尚未有持股，請先新增。")
        st.metric("已實現損益", f"{total_realized:,.4f}")
        _render_confirm_dialog()
        return

    rows: List[Dict[str, Any]] = []
    principal = 0.0
    total_value = 0.0
    links: List[Dict[str, str]] = []
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
        links.append({"代碼": sym, "前往": f"./?nav={'ETF' if guess_is_etf(sym) else '股票'}&symbol={sym}"})
        principal += cost * qty
        total_value += value

    df = pd.DataFrame(rows)
    try:
        df["_d"] = pd.to_datetime(df["買入日"], errors="coerce")
        df.sort_values(by=["_d", "代碼"], ascending=[True, True], inplace=True)
        df.drop(columns=["_d"], inplace=True)
    except Exception:
        pass

    def _pos_neg_color(v: Any) -> str:
        if isinstance(v, (int, float)) and pd.notna(v):
            if v > 0:
                return "color:green;"
            if v < 0:
                return "color:red;"
        return ""

    styled = (
        df.style
        .format(
            {
                "股數": "{:,.4f}",
                "成本/股": "{:,.4f}",
                "現價": "{:,.4f}",
                "市值": "{:,.4f}",
                "未實現損益": "{:,.4f}",
                "回報率%": "{:.2f}%",
            },
            na_rep="—",
        )
        .applymap(_pos_neg_color, subset=["未實現損益"])
        .applymap(_pos_neg_color, subset=["回報率%"])
    )
    st.dataframe(styled, use_container_width=True)

    # ======== 資產配置（市值占比）—— 恢復這一段 ========
    st.subheader("資產配置（市值占比）", anchor=False)
    alloc = (
        df[["代碼", "市值"]]
        .copy()
        .dropna(subset=["市值"])
        .groupby("代碼", as_index=False)["市值"]
        .sum()
        .sort_values("市值", ascending=False)
    )
    total_mv = alloc["市值"].sum() if not alloc.empty else 0.0
    if total_mv > 0:
        alloc["占比%"] = alloc["市值"] / total_mv * 100.0
        alloc_display = alloc.copy()
        alloc_display["市值"] = alloc_display["市值"].apply(lambda v: f"{v:,.4f}")
        alloc_display["占比%"] = alloc_display["占比%"].apply(lambda v: f"{v:.2f}%")
        st.dataframe(alloc_display, use_container_width=True, hide_index=True)
    else:
        st.info("目前無可用的市值資料。")

    # ======== 總結數據 ========
    pnl_unrealized = total_value - principal
    total_return_rate = (pnl_unrealized / principal * 100.0) if principal > 0 else 0.0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("總市值", f"{total_value:,.4f}")
        st.caption(f"本金：{principal:,.4f}")
    with c2:
        st.metric("總未實現損益", f"{pnl_unrealized:,.4f}", delta=f"{total_return_rate:.2f}%",
                  delta_color=("inverse" if pnl_unrealized < 0 else "normal"))
        st.caption(f"已實現損益：{sum(float(x.get('pnl', 0.0)) for x in _load_realized()):,.4f}")

    # ======== 管理持股 ========
    with st.expander("管理持股（刪除 / 賣出）", expanded=True):
        options = [f"{i+1}. {r.get('symbol')}｜買入日:{r.get('buy_date','—')}｜股數:{r.get('qty')}" for i, r in enumerate(data)]
        sel_idx = st.selectbox("選擇要操作的持股", options=range(len(options)), format_func=lambda i: options[i], key="mgmt_sel")
        cur = data[sel_idx]; cur_qty = int(cur.get("qty", 0))
        st.caption(f"目前選擇：{cur.get('symbol')}｜買入日 {cur.get('buy_date','—')}｜可用股數 {cur_qty:,}")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("刪除這筆持股", key="btn_delete", type="secondary"):
                st.session_state["confirm"] = {"type": "delete", "idx": sel_idx}
        with c2:
            sell_date_val = st.date_input("賣出日", value=date.today(), key="sell_date_global")
            sell_qty_val = st.number_input("賣出數量", min_value=1, max_value=max(cur_qty, 1),
                                           value=min(100, max(cur_qty, 1)), step=1, key="sell_qty_global")
        with c3:
            sell_price_val = st.number_input("賣出價格", min_value=0.0, value=0.0, step=0.0001, key="sell_price_global")
            if st.button("賣出", key="btn_sell", type="primary"):
                if float(sell_price_val) <= 0:
                    st.warning("請先輸入正確的賣出價格（>0）。")
                else:
                    st.session_state["confirm"] = {
                        "type": "sell",
                        "idx": sel_idx,
                        "sell_qty": int(sell_qty_val),
                        "sell_date": sell_date_val,
                        "sell_price": float(sell_price_val),
                    }

        st.divider()
        st.subheader("FIFO 賣出（依代碼跨批次）", anchor=False)
        symbols = sorted({str(r.get("symbol")) for r in data})
        fifo_symbol = st.selectbox("選擇代碼", options=symbols, key="fifo_sym")
        fifo_date = st.date_input("賣出日（FIFO）", value=date.today(), key="fifo_date")
        fifo_available = sum(int(r.get("qty", 0)) for r in data if str(r.get("symbol")) == fifo_symbol)
        c4, c5 = st.columns(2)
        with c4:
            fifo_price = st.number_input("賣出價格（FIFO）", min_value=0.0, value=0.0, step=0.0001, key="fifo_price")
        with c5:
            fifo_qty = st.number_input("賣出數量（FIFO）", min_value=1, max_value=max(fifo_available, 1),
                                       value=min(100, max(fifo_available, 1)), step=1, key="fifo_qty")
        st.caption(f"可用數量：{fifo_available:,}")
        if st.button("依 FIFO 賣出", type="primary", key="btn_fifo_sell"):
            if float(fifo_price) <= 0:
                st.warning("請先輸入正確的賣出價格（>0）。")
            else:
                st.session_state["confirm"] = {
                    "type": "sell_fifo",
                    "symbol": fifo_symbol,
                    "sell_qty": int(fifo_qty),
                    "sell_date": fifo_date,
                    "sell_price": float(fifo_price),
                }

    _render_confirm_dialog()
