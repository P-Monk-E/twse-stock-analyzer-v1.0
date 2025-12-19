# =========================================
# /mnt/data/stocks_page.py
# 修正：risk_grading.summarize 需傳 dict；完整改為分級摘要顯示
# 仍保留：60m/日/週/月 K、滑輪縮放、十字線、show() 相容 app.py
# =========================================
from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

from stock_utils import (
    find_ticker_by_name,
    get_metrics,
    is_etf as _is_etf_func,
    TICKER_NAME_MAP,
)
from chart_utils import plot_candlestick_with_ma, resample_ohlc, PLOTLY_TV_CONFIG
from risk_grading import (
    grade_alpha,
    grade_sharpe,
    grade_treynor,
    grade_debt_equity,
    grade_current_ratio,
    grade_roe,
    summarize,
)

# --------- 格式工具 ---------
def _fmt2(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.2f}"

def _fmt2pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x*100:.2f}%"

def _fmt0(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:,.0f}"

# --------- 市場／價格工具 ---------
def _normalize_tw_ticker_once(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW", ".TWO")) or s.startswith("^") else f"{s}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_intraday(ticker: str, interval: str = "60m", period: str = "60d") -> pd.DataFrame:
    """下載 60 分鐘線；錯誤時回空 df（避免整頁崩潰）。"""
    try:
        df = yf.Ticker(_normalize_tw_ticker_once(ticker)).history(period=period, interval=interval)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close"]].dropna(how="any")
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """alpha/beta 需市場收盤價；優先 TW 指數，不行用 GSPC。"""
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy()
                s.name = idx
                return s
        except Exception:
            continue
    return None

def _prepare_tf_df(ticker: str, base_daily_df: pd.DataFrame, tf_label: str) -> Tuple[pd.DataFrame, str]:
    """回傳 60m/日/週/月 dataframe 與標題附註。"""
    if tf_label == "60m":
        df = _download_ohlc_intraday(ticker, "60m", "60d")
        note = "（60 分鐘）"
    elif tf_label == "日":
        df = base_daily_df.copy()
        note = "（日 K）"
    elif tf_label == "週":
        df = resample_ohlc(base_daily_df, "W")
        note = "（週 K）"
    else:
        df = resample_ohlc(base_daily_df, "M")
        note = "（月 K）"
    return df, note

# --------- 主頁面 ---------
def render(prefill_symbol: Optional[str] = None) -> None:
    st.header("股票")
    col1, col2 = st.columns([3, 2])
    with col1:
        default_kw = prefill_symbol or st.session_state.get("last_stock_kw", "2330")
        keyword = st.text_input("輸入股票代碼或名稱", value=default_kw)
    with col2:
        tf_label = st.radio("K 線週期", options=["60m", "日", "週", "月"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：2330 或 台積電）")
        return

    try:
        ticker = find_ticker_by_name(keyword)  # 只回代碼
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_stock_kw"] = keyword

        # ---- get_metrics 需要的參數 ----
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=365)
        market_close = _get_market_close_series(start, end)
        if market_close is None:
            raise RuntimeError("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）")
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=False)

        # ======= KPI 區 =======
        with st.container(border=True):
            st.subheader(f"{name or ticker}（{ticker}）")
            if _is_etf_func(ticker):
                st.warning("這看起來像是 ETF，建議改到「ETF」分頁查詢。")

            # 依新規格：先做 dict，再 summarize(dict)
            grades = {
                "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
                "Treynor": grade_treynor(stats.get("Treynor")),
                "Alpha": grade_alpha(stats.get("Alpha")),
                "負債權益比": grade_debt_equity(stats.get("負債權益比")),
                "流動比率": grade_current_ratio(stats.get("流動比率")),
                "ROE": grade_roe(stats.get("ROE")),
            }
            crit, warn, good = summarize(grades)

            if crit:
                st.error("關鍵風險：" + "、".join(crit))
            if warn:
                st.warning("注意項：" + "、".join(warn))
            if good:
                st.success("達標：" + "、".join(good))

            equity = stats.get("Equity")
            eps_ttm = stats.get("EPS_TTM")
            v_roe = stats.get("ROE")
            st.markdown(
                f"**ROE**：{_fmt2pct(v_roe)} ｜ "
                f"**股東權益**：{_fmt0(equity)} ｜ "
                f"**EPS(TTM)**：{_fmt2(eps_ttm)}"
            )

        # ======= K 線 =======
        base_df: pd.DataFrame = stats["df"].copy()
        if not isinstance(base_df.index, pd.DatetimeIndex):
            base_df.index = pd.to_datetime(base_df.index)

        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf_label)
        if tf_df.empty:
            st.error("查無對應週期的價格資料。")
        else:
            title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
            fig = plot_candlestick_with_ma(tf_df, title=title, uirevision_key=f"{ticker}_{tf_label}")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

        madr = stats.get("MADR")
        st.caption(f"MADR：{madr:.4f}" if madr is not None and pd.notna(madr) else "MADR：—")

        # ======= 加入觀察 =======
        with st.container():
            right = st.columns([1, 1, 1, 1, 1, 1, 1])[-1]
            with right:
                if st.button("＋加入觀察", use_container_width=True):
                    from watchlist_page import add_to_watchlist
                    add_to_watchlist(symbol_or_name=ticker, name=name, kind_kw="stock")
                    st.success("已加入觀察名單")
    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")

# 與 app.py 相容的入口
def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
