# =========================================
# file: app.py
# =========================================
from __future__ import annotations

import streamlit as st
import stocks_page
import etf_page
import portfolio_page
import recommend_page

PAGES = ["股票", "ETF", "庫存", "推薦"]

def main():
    st.sidebar.header("主選單")
    nav = st.sidebar.radio("選擇頁面", PAGES, index=0, key="nav_page")

    # 為了返回時預填
    q_symbol = st.query_params.get("symbol")

    if nav == "股票":
        stocks_page.show(prefill_symbol=q_symbol)
    elif nav == "ETF":
        etf_page.show(prefill_symbol=q_symbol)
    elif nav == "庫存":
        portfolio_page.show(prefill_symbol=q_symbol)
    elif nav == "推薦":
        recommend_page.show()  # 推薦頁不需要 symbol

if __name__ == "__main__":
    main()


# =========================================
# file: stock_utils.py
# 共用工具：代碼/名稱查找、指標計算、.info 容錯
# =========================================
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# 可擴充
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
    return s  # fallback：使用者已輸入代碼

def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    try:
        return yf.Ticker(f"{code}.TW").history(start=start, end=end)
    except Exception:
        return None  # why: yfinance 有時回例外

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

    # 財報欄預設為 NaN（.info 常為 None/慢）
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan

    if not is_etf:
        try:
            info = yf.Ticker(f"{code}.TW").info or {}
            total_liab = info.get("totalLiab", np.nan)
            total_assets = info.get("totalAssets", np.nan)
            if not (np.isnan(total_liab) or np.isnan(total_assets)):
                equity = total_assets - total_liab
                debt_equity = (total_liab / equity) if equity != 0 else np.nan
            current_ratio = info.get("currentRatio", np.nan)
            roe = info.get("returnOnEquity", np.nan)
        except Exception:
            pass  # why: 容忍資料缺失，以❓呈現

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
    st.text_input("輸入股票名稱或代碼（例：台積電 或 2330）", value=default_symbol,
                  key="stock_symbol", on_change=_sync_symbol_from_input)
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
            st.metric("Alpha(年化)", value=f"{stats['Alpha']:.4f}" if stats["Alpha"] is not None else "—")
            st.caption(_tag(stats["Alpha"], 0, True) + " 越大越好")
        with col2:
            st.metric("Sharpe Ratio", value=f"{stats['Sharpe Ratio']:.2f}" if stats["Sharpe Ratio"] is not None else "—")
            st.caption(_tag(stats["Sharpe Ratio"], 1, True) + " >1 佳")
        with col3:
            st.metric("Beta", value=f"{stats['Beta']:.2f}" if stats["Beta"] is not None else "—")
            st.caption("相對市場波動")

        c1, c2, c3 = st.columns(3)
        v = stats.get("負債權益比"); c1.write(f"**負債權益比**：{v if pd.notna(v) else '—'} {_tag(v, 1, False)}")
        v = stats.get("流動比率");   c2.write(f"**流動比率**：{v if pd.notna(v) else '—'} {_tag(v, 1.5, True)}")
        v = stats.get("ROE");       c3.write(f"**ROE**：{(v*100):.2f}% {_tag(v, 0.10, True)}" if pd.notna(v) else "**ROE**：— ❓")

        fig = plot_candlestick_with_ma(stats["df"].copy(), title=f"{name or ticker}（{ticker}）技術圖")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"MADR：{stats['MADR']:.4f}" if stats["MADR"] is not None else "MADR：
