# /mnt/data/portfolio_page.py
# 📦 我的庫存（移除 pandas Styler 的 matplotlib 依賴；使用 st.data_editor 呈現）
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
REALIZED_PATH = "realized_trades.json"


def guess_is_etf(symbol: str) -> bool:
    s = symbol.upper().strip()
    return s.startswith("00") or s.startswith("009")


def get_latest_price(symbol: str) -> Optional[float]:
    """盡量取到最新價；先 fast_info，再退回 1d history。"""
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


# ---------- storage ----------
def _load_portfolio() -> List[Dict[str, Any]]:
    """
    向下相容讀檔：
    - 正常：list
    - 舊版：{"positions":[...]} → 轉成 list 並立即覆寫為純陣列
    """
    if "portfolio" in st.session_state and isinstance(st.session_state.portfolio, list):
        return st.session_state.portfolio
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                st.session_state.portfolio = data
            elif isinstance(data, dict) and isinstance(data.get("positions"), list):
                st.session_state.portfolio = data.get("positions", [])
                _save_portfolio()  # 立刻覆寫為純陣列
            else:
                st.session_state.portfolio = []
        except Exception:
            st.session_state.portfolio = []
    else:
        st.session_state.portfolio = []
    return st.session_state.portfolio


def _save_portfolio() -> None:
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state.portfolio, f, ensure_ascii=False, indent=2)


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


def _save_realized(rec: Dict[str, Any]) -> None:
    realized = _load_realized()
    realized.append(rec)
    with open(REALIZED_PATH, "w", encoding="utf-8") as f:
        json.dump(realized, f, ensure_ascii=False, indent=2)


# ---------- actions ----------
def _delete_position(idx: int) -> None:
    data = _load_portfolio()
    if 0 <= idx < len(data):
        data.pop(idx)
        _save_portfolio()
        st.success("已刪除。")
        st.rerun()


def _sell_position(idx: int, sell_qty: int, sell_date: date, sell_price: float) -> None:
    data = _load_portfolio()
    if 0 <= idx < len(data):
        pos = data[idx]
        qty = int(pos.get("qty", 0))
        sell_qty = min(sell_qty, qty)
        remain = qty - sell_qty
        realized_pnl = (sell_price - float(pos.get("cost", 0.0))) * sell_qty
        pos["qty"] = remain
        _save_portfolio()
        _save_realized(
            {
                "symbol": pos.get("symbol", ""),
                "sell_qty": sell_qty,
                "sell_date": sell_date.isoformat(),
                "sell_price": float(sell_price),
                "pnl": realized_pnl,
            }
        )
        st.success(f"已賣出 {sell_qty} 股，實現損益 {realized_pnl:,.2f}")
        st.rerun()


def _fifo_sell(symbol: str, sell_qty: int, sell_date: date, sell_price: float) -> None:
    data = _load_portfolio()
    sym = symbol.upper().strip()
    remain = sell_qty
    realized = []
    for pos in data:
        if pos.get("symbol") != sym or remain <= 0:
            continue
        take = min(int(pos.get("qty", 0)), remain)
        remain -= take
        pos["qty"] = int(pos.get("qty", 0)) - take
        realized.append((take, float(pos.get("cost", 0.0))))
    _save_portfolio()
    pnl = sum((sell_price - c) * q for q, c in realized)
    _save_realized(
        {
            "symbol": sym,
            "sell_qty": sell_qty,
            "sell_date": sell_date.isoformat(),
            "sell_price": float(sell_price),
            "pnl": pnl,
        }
    )
    st.success(f"FIFO 賣出完成，實現損益 {pnl:,.2f}")
    st.rerun()


def _render_confirm() -> None:
    if "confirm" not in st.session_state:
        return
    info = st.session_state.confirm
    st.warning(f"請確認操作：{info['msg']}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("確認", key="pf_ok"):
            try:
                t = info["type"]
                if t == "delete":
                    _delete_position(int(info["idx"]))
                elif t == "sell":
                    _sell_position(
                        int(info["idx"]),
                        int(info["sell_qty"]),
                        info["sell_date"],
                        float(info["sell_price"]),
                    )
                elif t == "sell_fifo":
                    _fifo_sell(
                        str(info["symbol"]),
                        int(info["sell_qty"]),
                        info["sell_date"],
                        float(info["sell_price"]),
                    )
            finally:
                st.session_state.pop("confirm", None)
                st.rerun()
    with c2:
        if st.button("取消", key="pf_cancel"):
            st.session_state.pop("confirm", None)
            st.info("已取消。")
            st.rerun()


# ---------- page ----------
def show(prefill_symbol: Optional[str] = None) -> None:
    st.header("📦 我的庫存")
    data = _load_portfolio()
    realized = _load_realized()

    # 新增持股（含分組）
    with st.expander("新增持股", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
        with c1:
            sym = st.text_input(
                "代碼（例：2330 或 2330.TW）", value=prefill_symbol or "", key="pf_add_sym"
            )
        with c2:
            qty = st.number_input("股數", min_value=1, value=100, step=100, key="pf_add_qty")
        with c3:
            cost = st.number_input("成本/股", min_value=0.0, value=100.0, step=0.1, key="pf_add_cost")
        with c4:
            buy_date = st.date_input("買入日", value=date.today(), key="pf_add_date")
        with c5:
            group = st.selectbox("分組", options=["", "防守型", "主力", "進攻型"], key="pf_add_group")
        if st.button("加入", type="primary"):
            if not sym.strip():
                st.warning("請輸入代碼。")
            else:
                data.append(
                    {
                        "symbol": sym.strip().upper(),
                        "qty": int(qty),
                        "cost": float(cost),
                        "buy_date": buy_date.isoformat(),
                        "group": group or "",
                    }
                )
                _save_portfolio()
                st.success("已加入。")
                st.rerun()

    # 已實現損益統計
    total_realized = sum(float(x.get("pnl", 0.0)) for x in realized)
    st.metric("已實現損益", f"{total_realized:,.4f}")

    if not data:
        st.info("目前尚未有持股，請先新增。")
        _render_confirm()
        return

    # 明細
    rows, links = [], []
    principal = 0.0
    total_value = 0.0
    for row in data:
        sym = row.get("symbol")
        qty = float(row.get("qty", 0.0))
        cost = float(row.get("cost", 0.0))
        price = get_latest_price(sym)
        value = (price or 0.0) * qty
        unreal = (price - cost) * qty if price is not None else None
        rate_pct = ((price - cost) / cost * 100.0) if (price is not None and cost > 0) else None
        rows.append(
            {
                "買入日": row.get("buy_date") or "—",
                "代碼": sym,
                "分組": row.get("group", ""),
                "股數": qty,
                "成本/股": cost,
                "現價": price,
                "市值": value,
                "未實現損益": unreal,
                "回報率%": rate_pct,
            }
        )
        nav = "ETF" if guess_is_etf(sym) else "股票"
        links.append({"代碼": sym, "前往": f"./?nav={nav}&symbol={sym}"})
        principal += cost * qty
        total_value += value

    df = pd.DataFrame(rows)

    # 用 data_editor 呈現（不使用 pandas Styler → 避免 matplotlib 依賴）
    st.subheader("持股明細", anchor=False)
    st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "買入日": st.column_config.TextColumn("買入日"),
            "代碼": st.column_config.TextColumn("代碼"),
            "分組": st.column_config.TextColumn("分組"),
            "股數": st.column_config.NumberColumn("股數", format="%.2f"),
            "成本/股": st.column_config.NumberColumn("成本/股", format="%.2f"),
            "現價": st.column_config.NumberColumn("現價", format="%.2f"),
            "市值": st.column_config.NumberColumn("市值", format="%.2f"),
            "未實現損益": st.column_config.NumberColumn("未實現損益", format="%.2f"),
            "回報率%": st.column_config.NumberColumn("回報率%", format="%.2f"),
        },
        key="pf_table",
    )

    st.caption("快速前往：")
    st.data_editor(
        pd.DataFrame(links),
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "代碼": st.column_config.TextColumn("代碼"),
            "前往": st.column_config.LinkColumn("前往專區"),
        },
    )

    # 風險評估（沿用你原本實作；不依賴 matplotlib）
    st.subheader("投組風險評估（示意）", anchor=False)
    try:
        sharpe, treynor, diff, dbg = estimate_portfolio_risk(
            df.rename(columns={"代碼": "symbol", "市值": "value"})
        )
        ca, cb, cc = st.columns(3)
        ca.metric("Sharpe", f"{(sharpe if sharpe is not None else float('nan')):.4f}")
        cb.metric("Treynor", f"{(treynor if treynor is not None else float('nan')):.4f}")
        cc.metric("Diff (T−S)", f"{(diff if diff is not None else float('nan')):.4f}")
        if sharpe is None and treynor is None:
            st.warning(f"⚠ 無法估算：{dbg}")
        elif treynor is None:
            st.warning(f"⚠ 僅估出 Sharpe，Treynor 無法估算：{dbg}")
        else:
            set_portfolio_risk_warning(sharpe, treynor, non_sys_thr=0.6, sys_thr=0.3)
            msg = diversification_warning(sharpe, treynor, non_sys_thr=0.6, sys_thr=0.3)
            st.warning(msg) if msg else st.success("✅ 未偵測到明顯分散/系統性風險失衡。")
    except Exception as e:
        st.caption(f"風險計算略過：{e}")

    _render_confirm()
