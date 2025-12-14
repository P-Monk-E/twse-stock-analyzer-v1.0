# =========================================
# file: app.py
# =========================================
from __future__ import annotations

import streamlit as st
import stocks_page
import etf_page
import portfolio_page

PAGES = ["股票", "ETF", "庫存"]

def main():
    st.sidebar.header("主選單")
    nav = st.sidebar.radio("選擇頁面", PAGES, index=0, key="nav_page")

    q_symbol = st.query_params.get("symbol")  # why: 返回同頁時預填

    if nav == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif nav == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif nav == "庫存":
        portfolio_page.show(prefill_symbol=q_symbol)

if __name__ == "__main__":
    main()


# =========================================
# file: stock_utils.py
# 共用工具與指標計算
# =========================================
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

TICKER_NAME_MAP = {
    "2330": "台積電",
    "2454": "聯發科",
    "2303": "聯電",
    "2618": "長榮航",
    "1737": "臺鹽",
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "006208": "富邦科技",
}
ETF_LIST = {"0050", "0056", "006208"}

def is_etf(code: str) -> bool:
    return code in ETF_LIST

def find_ticker_by_name(input_str: str) -> str:
    s = str(input_str).strip().upper()
    if s in TICKER_NAME_MAP:
        return s
    for t, name in TICKER_NAME_MAP.items():
        if s in name or s in name.upper():
            return t
    return s  # 假設使用者就輸入代碼

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None  # why: 避免 UI 中斷

def _calc_beta(prices_asset: pd.Series, prices_market: pd.Series) -> float | float("nan"):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    if df.empty:
        return np.nan
    df.columns = ["asset", "market"]
    am = df["asset"].resample("M").last().pct_change().dropna()
    mm = df["market"].resample("M").last().pct_change().dropna()
    if len(am) < 12:
        return np.nan
    X = sm.add_constant(mm)
    try:
        return float(sm.OLS(am, X).fit().params.get("market", np.nan))
    except Exception:
        return np.nan

def _calc_alpha(prices_asset: pd.Series, prices_market: pd.Series, rf: float) -> float | float("nan"):
    df = pd.concat([prices_asset, prices_market], axis=1).dropna()
    if df.empty:
        return np.nan
    ar = df.iloc[:, 0].pct_change().dropna()
    mr = df.iloc[:, 1].pct_change().dropna()
    if len(ar) == 0 or len(mr) == 0:
        return np.nan
    excess_a = ar - rf / 252
    excess_m = mr - rf / 252
    X = sm.add_constant(excess_m)
    try:
        return float(sm.OLS(excess_a, X).fit().params.get("const", np.nan)) * 252
    except Exception:
        return np.nan

def _calc_sharpe(prices: pd.Series, rf: float) -> float | float("nan"):
    r = prices.pct_change().dropna()
    if r.std() == 0 or len(r) == 0:
        return np.nan
    return ((r - rf / 252).mean() / r.std()) * np.sqrt(252)

def get_metrics(code: str, market_close: pd.Series, rf: float, start, end, is_etf: bool = False):
    df = fetch_price_data(code, start, end)
    if df is None or df.empty:
        return None

    close = df["Close"]
    beta = _calc_beta(close, market_close)
    alpha = _calc_alpha(close, market_close, rf)
    sharpe = _calc_sharpe(close, rf)

    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    if not is_etf:
        try:
            info = yf.Ticker(f"{code}.TW").info or {}
            tl = info.get("totalLiab", np.nan)
            ta = info.get("totalAssets", np.nan)
            if not (np.isnan(tl) or np.isnan(ta)):
                equity = ta - tl
                debt_equity = (tl / equity) if equity != 0 else np.nan
            current_ratio = info.get("currentRatio", np.nan)
            roe = info.get("returnOnEquity", np.nan)
        except Exception:
            pass  # why: yfinance info 常為 None/timeout

    returns = close.pct_change().dropna()
    madr = np.median(np.abs(returns - returns.mean())) if not returns.empty else np.nan

    return {
        "name": TICKER_NAME_MAP.get(code, ""),
        "負債權益比": debt_equity,
        "流動比率": current_ratio,
        "ROE": roe,
        "Alpha": alpha,
        "Sharpe Ratio": sharpe,
        "Beta": beta,
        "MADR": madr,
        "df": df,
    }


# =========================================
# file: stocks_page.py
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_ma

def _sync_symbol_from_input():
    txt = (st.session_state.get("stock_symbol") or "").strip()
    if txt:
        st.query_params["symbol"] = txt
    elif "symbol" in st.query_params:
        del st.query_params["symbol"]

def _tag(val: Optional[float], thr: float, greater: bool = True) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or pd.isna(val))):
        return "❓"
    return "✅" if ((val >= thr) if greater else (val <= thr)) else "❗"

def show(prefill_symbol: str | None = None) -> None:
    st.header("📈 股票專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入股票名稱或代碼（例：台積電 或 2330）",
                  value=default_symbol, key="stock_symbol", on_change=_sync_symbol_from_input)
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
        mkt_close = yf.Ticker("^TWII").history(start=start, end=end)["Close"]

        stats = get_metrics(ticker, mkt_close, rf, start, end, is_etf=False)
        if not stats:
            st.warning("查無該股票資料或資料不足。")
            return

        name = stats.get("name") or TICKER_NAME_MAP.get(ticker, "")
        st.subheader(f"{name or ticker}（{ticker}）")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha(年化)", f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        c1, c2, c3 = st.columns(3)
        v = stats.get("負債權益比"); c1.write(f"**負債權益比**：{v if pd.notna(v) else '—'} {_tag(v, 1, False)}")
        v = stats.get("流動比率");   c2.write(f"**流動比率**：{v if pd.notna(v) else '—'} {_tag(v, 1.5, True)}")
        v = stats.get("ROE");       c3.write(f"**ROE**：{(v*100):.2f}% {_tag(v, 0.10, True)}" if pd.notna(v) else "**ROE**：— ❓")

        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")


# =========================================
# file: etf_page.py
# =========================================
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
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
    return "✅" if ((val >= thr) if greater else (val <= thr)) else "❗"

def show(prefill_symbol: str | None = None) -> None:
    st.header("📊 ETF 專區")

    default_symbol = st.query_params.get("symbol", prefill_symbol or "")
    st.text_input("輸入 ETF 名稱或代碼（例：0050 / 0056 / 006208）",
                  value=default_symbol, key="etf_symbol", on_change=_sync_symbol_from_input)
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
            st.metric("Alpha(年化)", f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：—")

    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")


# =========================================
# file: portfolio_page.py
# =========================================
from __future__ import annotations

import json
import os
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


# =========================================
# file: tests/test_utils.py
# =========================================
from __future__ import annotations

import types
import numpy as np
import pandas as pd
import pytest

import portfolio_page as pf
import stock_utils as su

class _FakeTicker:
    def __init__(self, symbol, last=100.0, close=99.0, info=None, hist_days=30):
        self._last = last
        self._close = close
        self._info = info or {}
        self._hist_days = hist_days

    @property
    def fast_info(self):
        return {"lastPrice": self._last} if self._last is not None else {}

    def history(self, period=None, start=None, end=None):
        days = self._hist_days
        idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
        return pd.DataFrame({"Close": np.linspace(self._close - 1, self._close, days)}, index=idx)

    @property
    def info(self):
        return self._info

def _patch_yf(monkeypatch, last=None, close=99.0, info=None):
    def _factory(symbol):
        return _FakeTicker(symbol, last=last, close=close, info=info)
    yf_mod = types.SimpleNamespace(Ticker=_factory)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", yf_mod)

def test_get_latest_price_fast_info(monkeypatch):
    _patch_yf(monkeypatch, last=123.45)
    assert abs(pf.get_latest_price("2330") - 123.45) < 1e-6

def test_get_latest_price_history_fallback(monkeypatch):
    _patch_yf(monkeypatch, last=None, close=88.0)
    assert abs(pf.get_latest_price("2330") - 88.0) < 1e-6

def test_get_metrics_stock_info_guard(monkeypatch):
    idx = pd.date_range("2023-01-01", periods=30, freq="D")
    market = pd.Series(np.linspace(100, 110, 30), index=idx, name="Close")
    info = {"totalLiab": 50_000, "totalAssets": 100_000, "currentRatio": 1.8, "returnOnEquity": 0.12}
    _patch_yf(monkeypatch, last=100.0, close=100.0, info=info)

    def fake_fetch(code, start, end):
        return _FakeTicker(code, last=100.0, close=100.0).history(start=start, end=end)
    monkeypatch.setattr(su, "fetch_price_data", fake_fetch)

    m = su.get_metrics("2330", market, rf=0.01, start=idx[0], end=idx[-1], is_etf=False)
    assert m is not None and "Alpha" in m and "Sharpe Ratio" in m and "Beta" in m
    assert m["流動比率"] == pytest.approx(1.8, rel=1e-6)
    assert m["ROE"] == pytest.approx(0.12, rel=1e-6)

def test_get_metrics_etf_skips_info(monkeypatch):
    idx = pd.date_range("2023-01-01", periods=30, freq="D")
    market = pd.Series(np.linspace(100, 110, 30), index=idx, name="Close")
    _patch_yf(monkeypatch, last=100.0, close=100.0, info=None)

    def fake_fetch(code, start, end):
        return _FakeTicker(code, last=100.0, close=100.0).history(start=start, end=end)
    monkeypatch.setattr(su, "fetch_price_data", fake_fetch)

    m = su.get_metrics("0050", market, rf=0.01, start=idx[0], end=idx[-1], is_etf=True)
    assert m is not None
    import math
    assert math.isnan(m["負債權益比"]) and math.isnan(m["流動比率"]) and math.isnan(m["ROE"])
