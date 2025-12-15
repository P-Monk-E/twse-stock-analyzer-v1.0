# /mnt/data/watchlist_page.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import get_metrics  # 使用既有計算（股票/ETF KPI 與 ETF 的 EPS=配息TTM）

WATCHLIST_PATH = "watchlist.json"
PORTFOLIO_PATH = "portfolio.json"  # 與庫存頁相同檔名

# ========== Storage ==========
def _empty() -> Dict[str, List[Dict[str, Any]]]:
    return {"stocks": [], "etfs": []}

def load_watchlist() -> Dict[str, List[Dict[str, Any]]]:
    if "watchlist" in st.session_state and isinstance(st.session_state.watchlist, dict):
        wl = st.session_state.watchlist
    elif os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                wl = json.load(f)
        except Exception:
            wl = _empty()
    else:
        wl = _empty()
    for k in ("stocks", "etfs"):
        if k not in wl or not isinstance(wl[k], list):
            wl[k] = []
        for r in wl[k]:
            r.setdefault("pinned", False)
            r.setdefault("name", r.get("symbol", ""))
            r.setdefault("added_at", datetime.now().isoformat(timespec="seconds"))
            r["symbol"] = str(r.get("symbol", "")).upper()
    st.session_state.watchlist = wl
    return wl

def save_watchlist() -> None:
    if "watchlist" not in st.session_state: return
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.watchlist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {WATCHLIST_PATH} 失敗：{e}")

def add_to_watchlist(kind: str, symbol: str, name: str) -> None:
    wl = load_watchlist()
    key = "etfs" if kind == "etf" else "stocks"
    s = symbol.strip().upper()
    if any(s == x.get("symbol", "").upper() for x in wl[key]):
        st.info("已在觀察名單中。"); return
    wl[key].append({"symbol": s, "name": name or s, "added_at": datetime.now().isoformat(timespec="seconds"), "pinned": False})
    save_watchlist(); st.success("已加入觀察名單。")

def remove_from_watchlist(kind: str, symbols: List[str]) -> None:
    wl = load_watchlist()
    key = "etfs" if kind == "etf" else "stocks"
    to_del = {s.upper() for s in symbols}
    wl[key] = [r for r in wl[key] if r.get("symbol", "").upper() not in to_del]
    save_watchlist(); st.success("已刪除所選項目。")

# ========== Portfolio I/O（加入庫存用） ==========
def _load_portfolio() -> List[Dict[str, Any]]:
    if "portfolio" in st.session_state and isinstance(st.session_state.portfolio, list):
        return st.session_state.portfolio
    if os.path.exists(PORTFOLIO_PATH):
        try:
            with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                st.session_state.portfolio = json.load(f)
        except Exception:
            st.session_state.portfolio = []
    else:
        st.session_state.portfolio = []
    return st.session_state.portfolio

def _save_portfolio() -> None:
    try:
        with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.portfolio, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"寫入 {PORTFOLIO_PATH} 失敗：{e}")

# ========== KPI helpers ==========
@st.cache_data(ttl=1800)
def _fetch_metrics(symbol: str, is_etf: bool) -> Dict[str, Any]:
    try:
        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]
        stats = get_metrics(symbol, mkt_close, rf, start, end, is_etf=is_etf)
        return stats or {}
    except Exception:
        return {}

def _score(alpha: Any, sharpe: Any) -> float:
    try:
        a = float(alpha) if pd.notna(alpha) else 0.0
        s = float(sharpe) if pd.notna(sharpe) else 0.0
        if (alpha is None or pd.isna(alpha)) and (sharpe is None or pd.isna(sharpe)):
            return -1e12
        return 0.5 * a + 0.5 * s
    except Exception:
        return -1e12

def _fmt4(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)): return "—"
        return f"{float(x):.4f}"
    except Exception:
        return "—"

def _fmt2pct(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)): return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def _flag_fail_stock(alpha, sharpe, de, cr, roe) -> Dict[str, bool]:
    return {
        "Alpha": (pd.notna(alpha) and float(alpha) < 0),
        "Sharpe": (pd.notna(sharpe) and float(sharpe) < 1),
        "負債權益比": (pd.notna(de) and float(de) > 1),
        "流動比率": (pd.notna(cr) and float(cr) < 1.5),
        "ROE": (pd.notna(roe) and float(roe) < 0.15),
    }

def _flag_fail_etf(alpha, sharpe, treynor) -> Dict[str, bool]:
    return {
        "Alpha": (pd.notna(alpha) and float(alpha) < 0),
        "Sharpe": (pd.notna(sharpe) and float(sharpe) < 1),
        "Treynor": (pd.notna(treynor) and float(treynor) < 0),
    }

# ========== Confirm dialog ==========
def _render_confirm_dialog() -> None:
    info = st.session_state.get("wl_confirm")
    if not info: return
    with st.container():
        st.warning("請再次確認以下操作：")
        t = info.get("type")
        if t == "delete":
            st.write(f"刪除【{info.get('kind')}】：{', '.join(info.get('symbols', []))}")
        elif t == "add_portfolio":
            st.write(
                f"加入庫存：代碼 {info.get('symbol')}｜股數 {info.get('qty')}｜成本/股 {info.get('cost')}｜日期 {info.get('date')}"
            )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 確認", key="wl_confirm_yes"):
                try:
                    if t == "delete":
                        remove_from_watchlist(info.get("kind"), info.get("symbols", []))
                    elif t == "add_portfolio":
                        data = _load_portfolio()
                        data.append({
                            "symbol": info.get("symbol"),
                            "qty": int(info.get("qty")),
                            "cost": float(info.get("cost")),
                            "buy_date": info.get("date"),
                        })
                        _save_portfolio()
                        st.success("已加入庫存。")
                finally:
                    st.session_state.pop("wl_confirm", None)
                    st.rerun()
        with c2:
            if st.button("取消", key="wl_confirm_no"):
                st.session_state.pop("wl_confirm", None)
                st.info("已取消。")
                st.rerun()

# ========== UI section ==========
def _render_section(title: str, key_name: str, is_etf_list: bool) -> None:
    wl = load_watchlist()
    rows = wl[key_name]

    with st.expander(title, expanded=True):
        if not rows:
            st.info("目前沒有項目。"); return

        records: List[Dict[str, Any]] = []
        for r in rows:
            sym = r["symbol"].upper()
            stats = _fetch_metrics(sym, is_etf=is_etf_list)
            alpha = stats.get("Alpha"); sharpe = stats.get("Sharpe Ratio")
            beta = stats.get("Beta"); eps = stats.get("EPS_TTM")

            if is_etf_list:
                treynor = stats.get("Treynor")
                fails = _flag_fail_etf(alpha, sharpe, treynor)
                status_light = "🟡" if any(fails.values()) else ""
                records.append({
                    "釘選": "🟢" if r.get("pinned", False) else "",
                    "狀態": status_light,
                    "代碼": sym,
                    "名稱": r.get("name", sym),
                    "Alpha": _fmt4(alpha) + (" ❌" if fails["Alpha"] else ""),
                    "Sharpe": _fmt4(sharpe) + (" ❌" if fails["Sharpe"] else ""),
                    "Treynor": _fmt4(treynor) + (" ❌" if fails["Treynor"] else ""),
                    "Beta": _fmt4(beta),
                    "EPS(TTM)": _fmt4(eps),
                    "Score": _score(alpha, sharpe),
                    "前往": f"./?nav=ETF&symbol={sym}",
                })
            else:
                de = stats.get("負債權益比"); cr = stats.get("流動比率"); roe = stats.get("ROE")
                fails = _flag_fail_stock(alpha, sharpe, de, cr, roe)
                status_light = "🟡" if any(fails.values()) else ""
                records.append({
                    "釘選": "🟢" if r.get("pinned", False) else "",
                    "狀態": status_light,
                    "代碼": sym,
                    "名稱": r.get("name", sym),
                    "Alpha": _fmt4(alpha) + (" ❌" if fails["Alpha"] else ""),
                    "Sharpe": _fmt4(sharpe) + (" ❌" if fails["Sharpe"] else ""),
                    "Beta": _fmt4(beta),
                    "EPS(TTM)": _fmt4(eps),
                    "負債權益比": _fmt4(de) + (" ❌" if fails["負債權益比"] else ""),
                    "流動比率": _fmt4(cr) + (" ❌" if fails["流動比率"] else ""),
                    "ROE": _fmt2pct(roe) + (" ❌" if fails["ROE"] else ""),
                    "Score": _score(alpha, sharpe),
                    "前往": f"./?nav=股票&symbol={sym}",
                })

        df = pd.DataFrame(records)
        # 排序：釘選置頂（以是否空白表示）→ Score → 代碼
        df["pin_order"] = df["釘選"].apply(lambda x: 1 if x == "🟢" else 0)
        df.sort_values(by=["pin_order", "Score", "代碼"], ascending=[False, False, True], inplace=True, kind="mergesort")
        df.drop(columns=["pin_order"], inplace=True)

        # 顯示
        if is_etf_list:
            display_cols = ["釘選", "狀態", "代碼", "名稱", "Alpha", "Sharpe", "Treynor", "Beta", "EPS(TTM)", "Score", "前往"]
        else:
            display_cols = ["釘選", "狀態", "代碼", "名稱", "Alpha", "Sharpe", "Beta", "EPS(TTM)", "負債權益比", "流動比率", "ROE", "Score", "前往"]

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # —— 行為列：切換釘選 / 刪除（確認） / 加入庫存（確認） ——
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 2.6])

        with c1:
            target = st.selectbox("選擇代碼（切換釘選）", options=list(df["代碼"]), key=f"pin_sel_{key_name}")
            if st.button("切換釘選", key=f"btn_pin_{key_name}"):
                for r in st.session_state.watchlist[key_name]:
                    if r["symbol"].upper() == target:
                        r["pinned"] = not bool(r.get("pinned", False))
                        break
                save_watchlist(); st.success("釘選狀態已更新。"); st.rerun()

        with c2:
            to_delete = st.multiselect("選擇刪除", options=list(df["代碼"]), key=f"del_sel_{key_name}")
            if st.button("刪除選取", key=f"btn_del_{key_name}"):
                st.session_state["wl_confirm"] = {"type": "delete", "kind": ("etf" if is_etf_list else "stock"), "symbols": to_delete}

        with c3:
            # 加入庫存（一次一檔）
            add_sym = st.selectbox("加入庫存：代碼", options=list(df["代碼"]), key=f"add_sym_{key_name}")
            add_qty = st.number_input("股數", min_value=1, value=100, step=1, key=f"add_qty_{key_name}")
            add_cost = st.number_input("成本/股", min_value=0.0, value=0.0, step=0.0001, key=f"add_cost_{key_name}")
            add_date = st.date_input("買入日", value=date.today(), key=f"add_date_{key_name}")
        with c4:
            st.write("")  # 對齊
            st.write("")
            if st.button("＋ 加入庫存（需確認）", key=f"btn_add_{key_name}"):
                st.session_state["wl_confirm"] = {
                    "type": "add_portfolio",
                    "symbol": add_sym,
                    "qty": int(add_qty),
                    "cost": float(add_cost),
                    "date": add_date.isoformat(),
                }

        if st.button("重新整理 KPI（清快取）", key=f"btn_refresh_{key_name}"):
            _fetch_metrics.clear(); st.rerun()

# ========== Page entry ==========
def show() -> None:
    st.header("👀 觀察名單")
    load_watchlist()
    _render_section("待觀察股票", "stocks", is_etf_list=False)
    _render_section("待觀察ETF", "etfs", is_etf_list=True)
    _render_confirm_dialog()
