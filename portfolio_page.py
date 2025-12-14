# /mnt/data/portfolio_page.py
import base64
import csv
import io
import json
import os
from datetime import date
from typing import Any, Dict, List, Tuple

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
    # 清空則從 URL 刪除
    if value is None or str(value).strip() == "":
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
# CSV import / export helpers
# --------------------------
REQUIRED_FIELDS = ["ticker", "shares", "cost", "date"]

def _normalize_row(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """驗證與正規化一列；失敗回傳 (False, {}, reason)"""
    try:
        t = str(row.get("ticker", "")).upper().strip()
        sh = float(row.get("shares", 0))
        cost = float(row.get("cost", 0.0))
        d = str(row.get("date", "")).strip()
        if not t or sh < 0 or cost < 0:
            return False, {}, "欄位值不合法"
        return True, {"ticker": t, "shares": int(sh), "cost": float(cost), "date": d}, ""
    except Exception as e:
        return False, {}, f"資料格式錯誤：{e}"

def _portfolio_to_csv(portfolio: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REQUIRED_FIELDS)
    writer.writeheader()
    for p in portfolio:
        writer.writerow({
            "ticker": p.get("ticker", ""),
            "shares": p.get("shares", 0),
            "cost": p.get("cost", 0.0),
            "date": p.get("date", ""),
        })
    return buf.getvalue().encode("utf-8")

def _csv_bytes_to_rows(data: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    ok_rows, errors = [], []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8-sig", errors="ignore")
    rdr = csv.DictReader(io.StringIO(text))
    missing = [f for f in REQUIRED_FIELDS if f not in (rdr.fieldnames or [])]
    if missing:
        errors.append(f"缺少欄位: {', '.join(missing)}")
        return [], errors
    for i, raw in enumerate(rdr, start=1):
        valid, row, reason = _normalize_row(raw)
        if valid:
            ok_rows.append(row)
        else:
            errors.append(f"第 {i} 列錯誤：{reason}")
    return ok_rows, errors

def _merge_portfolio(base: List[Dict[str, Any]], new_rows: List[Dict[str, Any]], replace: bool) -> List[Dict[str, Any]]:
    idx = {p["ticker"].upper(): i for i, p in enumerate(base)}
    for r in new_rows:
        k = r["ticker"].upper()
        if k in idx:
            if replace:
                base[idx[k]] = r  # 覆蓋同代碼
        else:
            base.append(r)
    return base

# --------------------------
# Deletion confirmations
# --------------------------
def _init_delete_states():
    st.session_state.setdefault("pf_selected", set())
    st.session_state.setdefault("pf_confirm_single", None)
    st.session_state.setdefault("pf_confirm_batch", False)

# --------------------------
# Page
# --------------------------
def show():
    st.header("📦 庫存")

    portfolio = _load_portfolio()
    _init_delete_states()

    # ---- URL import (Base64 CSV) ----
    if "import" in st.query_params and not st.session_state.get("pf_import_done"):
        try:
            decoded = base64.b64decode(st.query_params["import"])
            rows, errs = _csv_bytes_to_rows(decoded)
            if errs:
                st.warning("URL 匯入錯誤：\n" + "\n".join(errs))
            else:
                portfolio[:] = _merge_portfolio(portfolio, rows, replace=False)
                _save_portfolio()
                st.success(f"已從 URL 匯入 {len(rows)} 筆。")
            st.session_state["pf_import_done"] = True  # 避免每次重整重覆匯入
        except Exception as e:
            st.warning(f"URL 匯入失敗：{e}")
            st.session_state["pf_import_done"] = True

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

    # ---- Import / Export (Sidebar) ----
    with st.sidebar.expander("匯入 / 匯出", expanded=True):
        # Export
        st.download_button(
            "⬇️ 匯出 CSV",
            data=_portfolio_to_csv(portfolio),
            file_name="portfolio.csv",
            mime="text/csv",
        )
        # Import (file)
        st.write("---")
        uploaded = st.file_uploader("上傳 CSV（欄位：ticker, shares, cost, date）", type=["csv"], accept_multiple_files=False)
        replace = st.checkbox("覆蓋相同代碼（勾選=覆蓋 / 未勾=合併）", value=False)
        if st.button("⬆️ 匯入檔案"):
            if not uploaded:
                st.warning("請先選擇檔案。")
            else:
                rows, errs = _csv_bytes_to_rows(uploaded.read())
                if errs:
                    st.error("匯入失敗：\n" + "\n".join(errs))
                else:
                    portfolio[:] = _merge_portfolio(portfolio, rows, replace=replace)
                    _save_portfolio()
                    st.success(f"匯入完成，共 {len(rows)} 筆。")
                    st.experimental_rerun()
        # Import (URL helper)
        st.caption("也可使用 URL 參數 `?import=<base64(csv)>` 自動匯入（一次性）。")

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

    # ---- Selection toolbar ----
    sel_cols = st.columns([1, 1])
    with sel_cols[0]:
        if st.button("全選顯示列"):
            st.session_state.pf_selected = {r["idx"] for r in rows}
    with sel_cols[1]:
        if st.button("清除選取"):
            st.session_state.pf_selected = set()

    # ---- Render table with selection & single delete ----
    for r in rows:
        i = r["idx"]
        selected = i in st.session_state.pf_selected
        c0, c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8])
        with c0:
            if st.checkbox("", value=selected, key=f"sel_{i}"):
                st.session_state.pf_selected.add(i)
            else:
                st.session_state.pf_selected.discard(i)
        with c1:
            st.markdown(f"**{r['代碼']}**")
            st.caption(r["date"])
        with c2:
            st.metric("持股", int(r["持股"]))
        with c3:
            st.metric("成本/股", round(r["成本"], 2))
        with c4:
            st.metric("現價", round(r["現價"], 2))
        with c5:
            st.metric("市值", round(r["市值"], 2))
        with c6:
            st.metric("報酬率(%)", round(r["報酬率"], 2))
        with c7:
            # 單筆刪除 → 先要求確認
            if st.button("🗑️", key=f"del_btn_{i}"):
                st.session_state.pf_confirm_single = i

    # ---- Batch delete trigger ----
    st.divider()
    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button(f"批次刪除（已選 {len(st.session_state.pf_selected)}）"):
            if not st.session_state.pf_selected:
                st.info("尚未勾選任何項目。")
            else:
                st.session_state.pf_confirm_batch = True

    # ---- Confirm blocks ----
    # Single confirm
    if st.session_state.pf_confirm_single is not None:
        i = st.session_state.pf_confirm_single
        st.warning(f"確認刪除索引 {i}？此動作無法還原。")
        c_yes, c_no = st.columns([1, 1])
        with c_yes:
            if st.button("✅ 確認刪除", key="confirm_single_yes"):
                try:
                    st.session_state.portfolio.pop(i)
                    _save_portfolio()
                    st.success("已刪除。")
                except Exception as e:
                    st.error(f"刪除失敗：{e}")
                finally:
                    st.session_state.pf_confirm_single = None
                    st.experimental_rerun()
        with c_no:
            if st.button("取消", key="confirm_single_no"):
                st.session_state.pf_confirm_single = None
                st.experimental_rerun()

    # Batch confirm
    if st.session_state.pf_confirm_batch:
        n = len(st.session_state.pf_selected)
        st.warning(f"確認批次刪除 {n} 筆？此動作無法還原。")
        c_yes, c_no = st.columns([1, 1])
        with c_yes:
            if st.button("✅ 確認批次刪除", key="confirm_batch_yes"):
                try:
                    to_del = sorted(list(st.session_state.pf_selected), reverse=True)  # 由大到小刪
                    for i in to_del:
                        if 0 <= i < len(st.session_state.portfolio):
                            st.session_state.portfolio.pop(i)
                    _save_portfolio()
                    st.success(f"已刪除 {len(to_del)} 筆。")
                except Exception as e:
                    st.error(f"批次刪除失敗：{e}")
                finally:
                    st.session_state.pf_selected = set()
                    st.session_state.pf_confirm_batch = False
                    st.experimental_rerun()
        with c_no:
            if st.button("取消", key="confirm_batch_no"):
                st.session_state.pf_confirm_batch = False
                st.experimental_rerun()

    # ---- Totals ----
    total_return = ((total_value - total_capital) / total_capital * 100) if total_capital > 0 else 0.0
    st.markdown(f"🔥 **總市值：{round(total_value,2)}**")
    st.markdown(f"💵 **總投入資金：{round(total_capital,2)}**")
    st.markdown(f"📉 **總報酬率：{round(total_return,2)}%**")
    st.caption(f"未實現損益：{round(total_unrealized,2)} 元")
    st.caption(f"🟩 已實現損益：{round(total_realized,2)} 元")
