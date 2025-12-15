# /mnt/data/watchlist_page.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, date
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import get_metrics  # 專案既有：計算 Alpha/Sharpe/Treynor/Beta/EPS(TTM)

WATCHLIST_PATH = "watchlist.json"
PORTFOLIO_PATH = "portfolio.json"

# 建議比例（中文）
RECO = {"防守型": 0.30, "主力": 0.50, "進攻型": 0.20}
TOL = 0.025  # 容許 ±2.5%

# =========================
# 儲存（觀察名單）
# =========================
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
            r["symbol"] = str(r.get("symbol", "")).upper()
            r.setdefault("name", r.get("symbol", ""))
            r.setdefault("added_at", datetime.now().isoformat(timespec="seconds"))
            r.setdefault("pinned", False)
            # 分組改為中文（防守型/主力/進攻型/空白）
            grp = r.get("group", "")
            if grp in ("defense", "core", "attack"):  # 舊值平滑升級
                r["group"] = {"defense": "防守型", "core": "主力", "attack": "進攻型"}[grp]
            else:
                r["group"] = str(grp)
    st.session_state.watchlist = wl
    return wl

def save_watchlist() -> None:
    if "watchlist" not in st.session_state:
        return
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
    wl[key].append({
        "symbol": s, "name": name or s,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "pinned": False, "group": ""
    })
    save_watchlist(); st.success("已加入觀察名單。")

def remove_from_watchlist(kind: str, symbols: List[str]) -> None:
    wl = load_watchlist()
    key = "etfs" if kind == "etf" else "stocks"
    to_del = {s.upper() for s in symbols}
    wl[key] = [r for r in wl[key] if r.get("symbol", "").upper() not in to_del]
    save_watchlist(); st.success("已刪除所選項目。")

# =========================
# 儲存（庫存，同步）
# =========================
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

# 取最新價（簡易容錯，30 分鐘快取）
@st.cache_data(ttl=1800)
def get_latest_price(symbol: str) -> float | None:
    s = symbol.strip().upper()
    cands = [s] if s.endswith((".TW", ".TWO")) else [f"{s}.TW", f"{s}.TWO"]
    for c in cands:
        try:
            p = yf.Ticker(c).fast_info.get("lastPrice")
            if p: return float(p)
        except Exception:
            pass
        try:
            hist = yf.Ticker(c).history(period="1d")
            if not hist.empty: return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return None

# 以「市值占比」計算三組目前配置（優先使用庫存中的分組，否則回退觀察名單分組）
def compute_group_allocation(wl: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    positions = _load_portfolio()
    if not positions:
        return {"防守型": 0.0, "主力": 0.0, "進攻型": 0.0}

    # 觀察名單分組 map（中文）
    wl_map: Dict[str, str] = {}
    for k in ("stocks", "etfs"):
        for r in wl.get(k, []):
            wl_map[str(r.get("symbol")).upper()] = str(r.get("group", ""))

    rows = []
    for p in positions:
        sym = str(p.get("symbol")).upper()
        qty = float(p.get("qty", 0))
        price = get_latest_price(sym) or 0.0
        mv = price * qty
        grp = str(p.get("group", "")).strip() or wl_map.get(sym, "")
        if grp in RECO and mv > 0:
            rows.append({"group": grp, "mv": mv})

    if not rows:
        return {"防守型": 0.0, "主力": 0.0, "進攻型": 0.0}

    df = pd.DataFrame(rows).groupby("group", as_index=False)["mv"].sum()
    total = float(df["mv"].sum())
    result = {"防守型": 0.0, "主力": 0.0, "進攻型": 0.0}
    if total <= 0:
        return result
    for _, r in df.iterrows():
        result[str(r["group"])] = float(r["mv"]) / total
    return result

# =========================
# KPI/排序/標示
# =========================
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
        if (alpha is None or pd.isna(alpha)) and (sharpe is None or pd.isna(sharpe)): return -1e12
        return 5.0 * a + 0.5 * s
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

def _fails_stock(alpha, sharpe, de, cr, roe) -> Dict[str, bool]:
    return {
        "Alpha": (pd.notna(alpha) and float(alpha) < 0),
        "Sharpe": (pd.notna(sharpe) and float(sharpe) < 1),
        "負債權益比": (pd.notna(de) and float(de) > 1),
        "流動比率": (pd.notna(cr) and float(cr) < 1.5),
        "ROE": (pd.notna(roe) and float(roe) < 0.15),
    }

def _fails_etf(alpha, sharpe, treynor) -> Dict[str, bool]:
    return {
        "Alpha": (pd.notna(alpha) and float(alpha) < 0),
        "Sharpe": (pd.notna(sharpe) and float(sharpe) < 1),
        "Treynor": (pd.notna(treynor) and float(treynor) < 0),
    }

# =========================
# 確認視窗：刪除／加入庫存
# =========================
def _render_confirm() -> None:
    info = st.session_state.get("wl_confirm")
    if not info: return
    st.warning("請再次確認以下操作：")
    if info["type"] == "delete":
        st.write(f"刪除【{ 'ETF' if info['kind']=='etf' else '股票' }】：{', '.join(info['symbols'])}")
    elif info["type"] == "add_portfolio":
        st.write(f"加入庫存：代碼 {info['symbol']}｜股數 {info['qty']}｜成本/股 {info['cost']}｜日期 {info['date']}｜分組 {info.get('group','') or '（未分組）'}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ 確認執行", key="wl_ok"):
            try:
                if info["type"] == "delete":
                    remove_from_watchlist(info["kind"], info["symbols"])
                else:
                    data = _load_portfolio()
                    data.append({
                        "symbol": info["symbol"],
                        "qty": int(info["qty"]),
                        "cost": float(info["cost"]),
                        "buy_date": info["date"],
                        "group": info.get("group", ""),  # 中文分組
                    })
                    _save_portfolio()
                    st.success("已加入庫存。")
            finally:
                st.session_state.pop("wl_confirm", None); st.rerun()
    with c2:
        if st.button("取消", key="wl_cancel"):
            st.session_state.pop("wl_confirm", None); st.info("已取消。"); st.rerun()

# =========================
# 觀察名單表格（釘選/刪除/連結/KPI）
# =========================
def _render_watch_table(kind_key: str, is_etf_list: bool) -> None:
    wl = load_watchlist()
    rows = wl[kind_key]
    if not rows:
        st.info("目前沒有項目。"); return

    out: List[Dict[str, Any]] = []
    for r in rows:
        sym = r["symbol"].upper()
        stats = _fetch_metrics(sym, is_etf=is_etf_list)
        alpha = stats.get("Alpha"); sharpe = stats.get("Sharpe Ratio")
        beta = stats.get("Beta"); eps = stats.get("EPS_TTM")
        score_val = _score(alpha, sharpe)

        if is_etf_list:
            trey = stats.get("Treynor")
            fails = _fails_etf(alpha, sharpe, trey); lamp = "🟡" if any(fails.values()) else ""
            out.append({
                "釘選": bool(r.get("pinned", False)),
                "狀態": lamp,
                "代碼": sym,
                "名稱": r.get("name", sym),
                "Alpha": _fmt4(alpha) + (" ❌" if fails["Alpha"] else ""),
                "Sharpe": _fmt4(sharpe) + (" ❌" if fails["Sharpe"] else ""),
                "Treynor": _fmt4(trey) + (" ❌" if fails["Treynor"] else ""),
                "Beta": _fmt4(beta),
                "EPS(TTM)": _fmt4(eps),
                "Score": _fmt4(score_val),
                "前往": f"./?nav=ETF&symbol={sym}",
                "🗑 刪除": False,
            })
        else:
            de = stats.get("負債權益比"); cr = stats.get("流動比率"); roe = stats.get("ROE")
            fails = _fails_stock(alpha, sharpe, de, cr, roe); lamp = "🟡" if any(fails.values()) else ""
            out.append({
                "釘選": bool(r.get("pinned", False)),
                "狀態": lamp,
                "代碼": sym,
                "名稱": r.get("name", sym),
                "Alpha": _fmt4(alpha) + (" ❌" if fails["Alpha"] else ""),
                "Sharpe": _fmt4(sharpe) + (" ❌" if fails["Sharpe"] else ""),
                "Beta": _fmt4(beta),
                "EPS(TTM)": _fmt4(eps),
                "負債權益比": _fmt4(de) + (" ❌" if fails["負債權益比"] else ""),
                "流動比率": _fmt4(cr) + (" ❌" if fails["流動比率"] else ""),
                "ROE": _fmt2pct(roe) + (" ❌" if fails["ROE"] else ""),
                "Score": _fmt4(score_val),
                "前往": f"./?nav=股票&symbol={sym}",
                "🗑 刪除": False,
            })

    df = pd.DataFrame(out)
    # 排序：釘選置頂 → Score 高 → 代碼
    def _to_float(s: str) -> float:
        try: return float(s)
        except Exception: return -1e12
    df["pin_order"] = df["釘選"].apply(lambda x: 1 if x else 0)
    df["score_order"] = df["Score"].apply(_to_float)
    df.sort_values(by=["pin_order", "score_order", "代碼"], ascending=[False, False, True], inplace=True, kind="mergesort")
    df.drop(columns=["pin_order", "score_order"], inplace=True)

    column_config = {
        "釘選": st.column_config.CheckboxColumn("釘選", help="切換釘選（變更即自動存檔）"),
        "狀態": st.column_config.TextColumn("狀態", disabled=True),
        "代碼": st.column_config.TextColumn("代碼", disabled=True),
        "名稱": st.column_config.TextColumn("名稱", disabled=True),
        "Alpha": st.column_config.TextColumn("Alpha"),
        "Sharpe": st.column_config.TextColumn("Sharpe"),
        "Beta": st.column_config.TextColumn("Beta"),
        "EPS(TTM)": st.column_config.TextColumn("EPS(TTM)"),
        "Score": st.column_config.TextColumn("Score", help="Score = 5×Alpha + 0.5×Sharpe"),
        "前往": st.column_config.LinkColumn("前往"),
        "🗑 刪除": st.column_config.CheckboxColumn("🗑 刪除", help="勾選後按下方『刪除選取』"),
    }
    if is_etf_list:
        column_config["Treynor"] = st.column_config.TextColumn("Treynor")
        display_cols = ["釘選","狀態","代碼","名稱","Alpha","Sharpe","Treynor","Beta","EPS(TTM)","Score","前往","🗑 刪除"]
    else:
        for col in ("負債權益比","流動比率","ROE"):
            column_config[col] = st.column_config.TextColumn(col)
        display_cols = ["釘選","狀態","代碼","名稱","Alpha","Sharpe","Beta","EPS(TTM)","負債權益比","流動比率","ROE","Score","前往","🗑 刪除"]

    edited = st.data_editor(
        df[display_cols], use_container_width=True, hide_index=True,
        column_config=column_config, key=f"editor_{kind_key}",
    )

    # 釘選變更 → 自動存檔
    new_pin_map = {row["代碼"]: bool(row["釘選"]) for _, row in edited.iterrows()}
    changed = False
    for r in st.session_state.watchlist[kind_key]:
        new_pin = new_pin_map.get(r["symbol"].upper(), r.get("pinned", False))
        if bool(r.get("pinned", False)) != bool(new_pin):
            r["pinned"] = bool(new_pin); changed = True
    if changed:
        save_watchlist(); st.caption("✅ 釘選變更已自動存檔。")

    # 刪除
    to_delete = [row["代碼"] for _, row in edited.iterrows() if bool(row["🗑 刪除"])]
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("刪除選取", key=f"btn_del_{kind_key}", type="secondary", disabled=(len(to_delete) == 0)):
            st.session_state["wl_confirm"] = {"type":"delete","kind":("etf" if is_etf_list else "stock"),"symbols":to_delete}
    with c2:
        if st.button("重新整理 KPI（清快取）", key=f"btn_refresh_{kind_key}"):
            _fetch_metrics.clear(); st.rerun()

# =========================
# 快速存取 + 分組（中文） + 資金配置線條圖
# =========================
def _render_quick_and_groups() -> None:
    st.subheader("⚡ 快速存取（同步庫存）與策略分組 / 資金配置", anchor=False)
    wl = load_watchlist()

    # 快速存取（含分組）
    all_syms = [r["symbol"] for k in ("stocks","etfs") for r in wl[k]]
    c1, c2, c3, c4, c5, c6 = st.columns([1.2,1,1,1,1,1.4])
    with c1:
        sel = st.selectbox("股票/ETF", options=all_syms, key="qa_sym")
    with c2:
        qty = st.number_input("股數", min_value=1, value=100, step=1, key="qa_qty")
    with c3:
        cost = st.number_input("成本/股", min_value=0.0, value=0.0, step=0.0001, key="qa_cost")
    with c4:
        dt = st.date_input("日期", value=date.today(), key="qa_date")
    with c5:
        grp = st.selectbox("分組", options=["", "防守型", "主力", "進攻型"], key="qa_group")
    with c6:
        if st.button("＋ 快速存取（需確認）", key="qa_btn"):
            st.session_state["wl_confirm"] = {
                "type":"add_portfolio","symbol":sel,"qty":int(qty),
                "cost":float(cost),"date":dt.isoformat(),"group":grp
            }

    st.markdown("---")

    # 分組編輯表（中文，下拉、即時存檔）
    st.caption("分組設定（中文）：防守型 / 主力 / 進攻型")
    rows = []
    for k in ("stocks","etfs"):
        for r in wl[k]:
            rows.append({"代碼": r["symbol"], "名稱": r["name"], "分組": r.get("group","")})
    gdf = pd.DataFrame(rows)
    edited = st.data_editor(
        gdf, hide_index=True, use_container_width=True,
        column_config={
            "代碼": st.column_config.TextColumn("代碼", disabled=True),
            "名稱": st.column_config.TextColumn("名稱", disabled=True),
            "分組": st.column_config.SelectboxColumn("分組", options=["","防守型","主力","進攻型"]),
        },
        key="group_editor",
    )
    group_map = {r["代碼"]: r["分組"] for _, r in edited.iterrows()}
    changed = False
    for k in ("stocks","etfs"):
        for r in st.session_state.watchlist[k]:
            newg = group_map.get(r["symbol"], r.get("group",""))
            if str(newg) != str(r.get("group","")):
                r["group"] = str(newg); changed = True
    if changed:
        save_watchlist(); st.caption("✅ 分組變更已自動存檔。")

    # 目前配置（依庫存市值）與建議配置 → 線條圖 + 文本差值 + 警告
    alloc = compute_group_allocation(st.session_state.watchlist)  # {'防守型':x, '主力':y, '進攻型':z}
    idx = ["防守型","主力","進攻型"]
    df_chart = pd.DataFrame({
        "建議": [RECO[i]*100 for i in idx],
        "目前": [alloc[i]*100 for i in idx],
    }, index=idx)
    st.line_chart(df_chart, use_container_width=True)  # 兩條線：建議/目前

    def _line(name: str) -> str:
        cur = alloc[name]; reco = RECO[name]; diff = cur - reco
        tag = ""
        if cur < (reco - TOL): tag = "（不足）"
        elif cur > (reco + TOL): tag = "（過多）"
        return f"**{name}**：建議 {reco*100:.1f}%｜目前 {cur*100:.2f}%｜差 {diff*100:+.2f}% {tag}"
    st.info("｜".join([_line(x) for x in idx]))

    warns = []
    if alloc["防守型"] < RECO["防守型"] - TOL: warns.append("防守型不足")
    if alloc["主力"] > RECO["主力"] + TOL:     warns.append("主力過多")
    if alloc["進攻型"] > RECO["進攻型"] + TOL: warns.append("進攻型過多")
    if warns:
        st.warning("；".join(warns) + "。")

# =========================
# Page entry
# =========================
def show() -> None:
    st.header("👀 觀察名單")
    load_watchlist()

    tab_stock, tab_etf = st.tabs(["待觀察股票", "待觀察ETF"])
    with tab_stock:
        _render_watch_table("stocks", is_etf_list=False)
    with tab_etf:
        _render_watch_table("etfs", is_etf_list=True)

    _render_quick_and_groups()
    _render_confirm()
