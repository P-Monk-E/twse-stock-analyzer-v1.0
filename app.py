# =========================================
# file: app.py
# （相容原有呼叫；若你不想改 app.py，可直接忽略本檔。）
# =========================================
import streamlit as st
import stocks_page
import etf_page
import portfolio_page

PAGES = ["股票", "ETF", "庫存"]

def main():
    st.sidebar.header("主選單")
    nav = st.sidebar.radio("選擇頁面", PAGES, index=0, key="nav_page")

    # 讀取 ?symbol= 供各頁預填（可為 None）
    q_symbol = st.query_params.get("symbol")

    if nav == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif nav == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif nav == "庫存":
        portfolio_page.show(prefill_symbol=q_symbol)

if __name__ == "__main__":
    main()


# =========================================
# file: stocks_page.py
# 股票頁：代碼/名稱 → K 線 + 指標 + 財報欄（容錯）
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import (
    find_ticker_by_name,
    get_metrics,
    is_etf,
    TICKER_NAME_MAP,
)
from chart_utils import plot_candlestick_with_ma


def _sync_symbol_from_input():
    """為了在換頁返回時自動帶入；避免殘留空字串。"""
    txt = (st.session_state.get("stock_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]


def _tag(val: Optional[float], thr: float, greater: bool = True) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    good = (val >= thr) if greater else (val <= thr)
    return "✅" if good else "❗"


def show(prefill_symbol: str | None = None) -> None:
    st.header("📈 股票專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input(
        "輸入股票名稱或代碼（例：台積電 或 2330）",
        value=default_symbol,
        key="stock_symbol",
        on_change=_sync_symbol_from_input,
    )
    user_input = (st.session_state.get("stock_symbol") or "").strip()
    if not user_input:
        st.info("請輸入股票名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
        if is_etf(ticker):
            st.warning("偵測到輸入為 ETF，請切換至「ETF」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01

        # 市場：加權指數（台灣）
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=False)
        if not stats:
            st.warning("查無該股票資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        st.subheader(f"{name or ticker}（{ticker}）")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", value=f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, greater=True) + " 越大越好（>0）")  # why: 讓使用者一眼判讀
        with col2:
            st.metric("Sharpe Ratio", value=f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, greater=True) + " 風險調整後報酬（>1 佳）")
        with col3:
            st.metric("Beta", value=f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        # 財報三欄（若取不到以❓呈現）
        c1, c2, c3 = st.columns(3)
        with c1:
            v = stats.get("負債權益比")
            st.write(f"**負債權益比**：{v if pd.notna(v) else '—'} {_tag(v, 1, greater=False)}")
        with c2:
            v = stats.get("流動比率")
            st.write(f"**流動比率**：{v if pd.notna(v) else '—'} {_tag(v, 1.5, greater=True)}")
        with c3:
            v = stats.get("ROE")
            st.write(f"**ROE**：{(v*100):.2f}% {_tag(v, 0.10, greater=True)}" if pd.notna(v) else "**ROE**：— ❓")

        # 技術圖
        df = stats["df"]
        fig = plot_candlestick_with_ma(df.copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)

        # 額外：MADR
        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")


# =========================================
# file: etf_page.py
# ETF 頁：代碼/名稱 → K 線 + 指標（不抓財報欄）
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import (
    find_ticker_by_name,
    get_metrics,
    is_etf,
    TICKER_NAME_MAP,
)
from chart_utils import plot_candlestick_with_ma


def _sync_symbol_from_input():
    txt = (st.session_state.get("etf_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]


def _tag(val: Optional[float], thr: float, greater: bool = True) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    good = (val >= thr) if greater else (val <= thr)
    return "✅" if good else "❗"


def show(prefill_symbol: str | None = None) -> None:
    st.header("📊 ETF 專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input(
        "輸入 ETF 名稱或代碼（例：0050 / 0056 / 006208）",
        value=default_symbol,
        key="etf_symbol",
        on_change=_sync_symbol_from_input,
    )
    user_input = (st.session_state.get("etf_symbol") or "").strip()
    if not user_input:
        st.info("請輸入 ETF 名稱或代碼以查詢。")
        return

    try:
        ticker = find_ticker_by_name(user_input)
        if not is_etf(ticker):
            st.warning("偵測到輸入為個股，請切換至「股票」頁面查詢。")
            return

        end = datetime.today()
        start = end - timedelta(days=365 * 3)
        rf = 0.01
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=True)
        if not stats:
            st.warning("查無 ETF 資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        st.subheader(f"{name or ticker}（{ticker}）")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", value=f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, greater=True) + " 越大越好（>0）")
        with col2:
            st.metric("Sharpe Ratio", value=f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, greater=True) + " 風險調整後報酬（>1 佳）")
        with col3:
            st.metric("Beta", value=f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        df = stats["df"]
        fig = plot_candlestick_with_ma(df.copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")


# =========================================
# file: portfolio_page.py
# 庫存頁：統一 show 簽名；穩健取價；其餘維持簡潔示範
# =========================================
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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
        # 先嘗試 fast_info
        try:
            info = yf.Ticker(c).fast_info
            p = info.get("lastPrice")
            if p:
                return float(p)
        except Exception:
            pass
        # 後援：用 history() 拿最後收盤
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

    with st.expander("新增持股"):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sym = st.text_input("代碼（例：2330 或 2330.TW）", value=prefill_symbol or "", key="pf_add_sym")
        with c2:
            qty = st.number_input("股數", min_value=1, value=100, step=100, key="pf_add_qty")
        with c3:
            cost = st.number_input("成本/股", min_value=0.0, value=100.0, step=0.1, key="pf_add_cost")
        if st.button("加入"):
            if not sym.strip():
                st.warning("請輸入代碼。")
            else:
                data.append({"symbol": sym.strip(), "qty": int(qty), "cost": float(cost)})
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
        sym = row["symbol"]
        qty = float(row["qty"])
        cost = float(row["cost"])
        price = get_latest_price(sym)
        value = (price or 0.0) * qty
        rows.append(
            {
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

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    pnl = total_value - total_cost
    st.metric("總市值", f"{total_value:,.0f}")
    st.metric("總損益", f"{pnl:,.0f}")

