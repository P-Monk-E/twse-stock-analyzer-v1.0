# /mnt/data/watchlist_page.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from stock_utils import get_metrics, is_etf as _is_etf
from names_store import get as get_name_override, set as set_name_override

WATCHLIST_PATH = "watchlist.json"
PORTFOLIO_PATH = "portfolio.json"

# ---------- watchlist storage ----------
def _ensure_state():
    if "watchlist" not in st.session_state or not isinstance(st.session_state.watchlist, dict):
        st.session_state.watchlist = {"stocks": [], "etfs": []}
    if "KPI_VERSION" not in st.session_state:
        st.session_state.KPI_VERSION = 1

def load_watchlist() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_state()
    if os.path.exists(WATCHLIST_PATH) and not st.session_state.watchlist.get("_loaded"):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {"stocks": [], "etfs": []}
            data.setdefault("stocks", [])
            data.setdefault("etfs", [])
            st.session_state.watchlist = data
            st.session_state.watchlist["_loaded"] = True
        except Exception:
            st.session_state.watchlist = {"stocks": [], "etfs": [], "_loaded": True}
    return st.session_state.watchlist

def save_watchlist() -> None:
    _ensure_state()
    data = st.session_state.watchlist.copy()
    data.pop("_loaded", None)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- public API ----------
def add_to_watchlist(kind: str, symbol: str, name: Optional[str] = None) -> None:
    load_watchlist()
    sym = (symbol or "").strip().upper()
    if not sym:
        st.warning("缺少代碼")
        return
    kind_key = "etfs" if (kind == "etf" or _is_etf(sym) or sym.startswith("00")) else "stocks"
    target = st.session_state.watchlist[kind_key]
    if any((r.get("symbol","").upper()==sym) for r in target):
        st.info("已在觀察名單中。"); return
    nm = (name or get_name_override(sym, sym)).strip()
    target.append({"symbol": sym, "name": nm or sym, "pinned": False, "added_at": datetime.now().isoformat(timespec="seconds")})
    save_watchlist()
    try: set_name_override(sym, nm or sym)
    except Exception: pass
    st.success(f"已加入觀察：{sym}")

# ---------- KPI cache (versioned) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def _metrics_for(sym: str, is_etf: bool, version: int) -> Dict[str, Any]:
    try:
        stats = get_metrics(sym, None, None, None, None, is_etf=is_etf)
        return {} if not stats else {
            "Alpha": stats.get("Alpha"),
            "Sharpe": stats.get("Sharpe"),
            "Treynor": stats.get("Treynor"),
            "Beta": stats.get("Beta"),
            "EPS_TTM": stats.get("EPS_TTM"),
            "Score": _calc_score(stats.get("Alpha"), stats.get("Sharpe")),
        }
    except Exception:
        return {}

def _calc_score(alpha: Optional[float], sharpe: Optional[float]) -> float:
    a = 0.0 if (alpha is None or (isinstance(alpha, float) and np.isnan(alpha))) else float(alpha)
    s = 0.0 if (sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe))) else float(sharpe)
    return 5 * a + 0.5 * s

def _refresh_metrics():
    st.session_state.KPI_VERSION += 1
    st.toast("KPI 已重新整理")
    st.rerun()

# ---------- portfolio I/O (純陣列；向下相容讀 dict.positions) ----------
def _load_portfolio() -> List[Dict[str, Any]]:
    if os.path.exists(PORTFOLIO_PATH):
        try:
            with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("positions"), list):
                return list(data.get("positions", []))  # 相容舊檔
        except Exception:
            pass
    return []

def _save_portfolio(lst: List[Dict[str, Any]]) -> None:
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

def _quick_add_position(symbol: str, name: str, qty: float, cost: float, group: str) -> None:
    data = _load_portfolio()
    data.append({
        "symbol": symbol.upper(),
        "qty": float(qty),
        "cost": float(cost),
        "buy_date": datetime.now().strftime("%Y-%m-%d"),
        "group": group or "",
        "name": name,
    })
    _save_portfolio(data)
    st.toast("已寫入庫存（已使用純陣列格式）")

# ---------- table render ----------
def _render_watch_table(kind_key: str, is_etf_list: bool) -> None:
    load_watchlist()
    data = st.session_state.watchlist.get(kind_key, [])
    if not data:
        st.info("清單為空。請至股票/ETF頁面點「＋加入觀察」，或在下方新增。")
        return

    rows = []
    ver = st.session_state.KPI_VERSION
    for r in data:
        sym = r.get("symbol", "").upper()
        name = get_name_override(sym, r.get("name", sym))
        met = _metrics_for(sym, is_etf_list, ver)
        rows.append({
            "釘選": bool(r.get("pinned", False)),
            "代碼": sym,
            "名稱": name,
            "Alpha": met.get("Alpha"),
            "Sharpe": met.get("Sharpe"),
            "Treynor": met.get("Treynor") if is_etf_list else None,
            "Beta": met.get("Beta"),
            "EPS(TTM)": met.get("EPS_TTM"),
            "Score": met.get("Score"),
            "快速前往": f"./?nav={'ETF' if is_etf_list else '股票'}&symbol={sym}",
        })

    df = pd.DataFrame(rows).sort_values(by=["釘選","Score"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    column_config = {
        "釘選": st.column_config.CheckboxColumn("釘選"),
        "代碼": st.column_config.TextColumn("代碼"),
        "名稱": st.column_config.TextColumn("名稱"),
        "Alpha": st.column_config.NumberColumn("Alpha", format="%.2f"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "Treynor": st.column_config.NumberColumn("Treynor", format="%.2f") if is_etf_list else None,
        "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        "EPS(TTM)": st.column_config.NumberColumn("EPS(TTM)", format="%.2f"),
        "Score": st.column_config.NumberColumn("Score", format="%.2f"),
        "快速前往": st.column_config.LinkColumn("快速前往"),
    }
    display_cols = [c for c in ["釘選","代碼","名稱","Alpha","Sharpe","Treynor","Beta","EPS(TTM)","Score","快速前往"] if c in df.columns]
    edited = st.data_editor(
        df[display_cols], use_container_width=True, hide_index=True,
        column_config={k:v for k,v in column_config.items() if v is not None},
        key=f"editor_{kind_key}",
    )

    pin_map  = {row["代碼"]: bool(row["釘選"]) for _, row in edited.iterrows()}
    name_map = {row["代碼"]: str(row["名稱"]).strip() for _, row in edited.iterrows()}

    changed = False
    for r in st.session_state.watchlist[kind_key]:
        sym = r.get("symbol", "").upper()
        if pin_map.get(sym, False) != r.get("pinned", False):
            r["pinned"] = pin_map.get(sym, False); changed = True
        new_name = name_map.get(sym)
        if new_name and new_name != r.get("name", sym):
            r["name"] = new_name; changed = True
            try: set_name_override(sym, new_name)
            except Exception: pass

    if changed:
        save_watchlist()
        st.caption("✅ 名稱/釘選變更已存檔並同步。")

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("重新整理 KPI", key=f"refresh_{kind_key}"):
            _refresh_metrics()
    with c2:
        to_delete = st.text_input("刪除代碼（逗號分隔）", key=f"del_{kind_key}")
        if st.button("刪除選取", key=f"btn_del_{kind_key}"):
            dels = {s.strip().upper() for s in to_delete.split(",") if s.strip()}
            st.session_state.watchlist[kind_key] = [r for r in st.session_state.watchlist[kind_key] if r.get("symbol","").upper() not in dels]
            save_watchlist(); st.rerun()
    with c3:
        st.caption("提示：直接在表格編輯「名稱」，關閉編輯即自動比較並儲存。")

# ---------- quick access ----------
def _render_quick_and_alloc() -> None:
    st.subheader("⚡ 快速存取")
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    sym = c1.text_input("代碼", key="qa_sym").upper()
    name = c2.text_input("名稱（可留空）", key="qa_name")
    qty = c3.number_input("股數/張數", min_value=0.0, value=0.0, step=1.0, key="qa_qty")
    cost = c4.number_input("成本", min_value=0.0, value=0.0, step=0.1, key="qa_cost")
    group = c5.selectbox("分組", ["防守型", "主力", "進攻型"], key="qa_group")

    if st.button("寫入庫存", key="qa_btn"):
        nm = name.strip() or get_name_override(sym, sym)
        _quick_add_position(sym, nm, qty, cost, group)
        try: set_name_override(sym, nm)
        except Exception: pass

    st.subheader("💰 資金配置概覽")
    alloc = {"防守型": 40, "主力": 40, "進攻型": 20}
    st.progress(min(100, alloc["防守型"]), text=f"防守型 {alloc['防守型']}%")
    st.progress(min(100, alloc["主力"]), text=f"主力 {alloc['主力']}%")
    st.progress(min(100, alloc["進攻型"]), text=f"進攻型 {alloc['進攻型']}%")
    RECO = {"防守型": 40, "主力": 40, "進攻型": 20}; TOL  = 5
    warns = []
    if alloc["防守型"] < RECO["防守型"]-TOL: warns.append("防守型不足")
    if alloc["主力"]   > RECO["主力"]  +TOL: warns.append("主力過多")
    if alloc["進攻型"] > RECO["進攻型"]+TOL: warns.append("進攻型過多")
    if warns: st.warning("；".join(warns)+"。")

def show() -> None:
    st.header("👀 觀察名單")
    load_watchlist()
    tab_stock, tab_etf = st.tabs(["待觀察股票", "待觀察ETF"])
    with tab_stock: _render_watch_table("stocks", is_etf_list=False)
    with tab_etf:   _render_watch_table("etfs",   is_etf_list=True)
    _render_quick_and_alloc()
    with st.expander("操作確認/歷史紀錄", expanded=False):
        st.caption("此區預留擴充：顯示最近操作歷程與確認訊息。")
