# =========================================
# /mnt/data/stocks_page.py
# 1) 僅保留「日K」；2) 嚴格分流（ETF 轉去 ETF 分頁）；3) KPI 數據區完整顯示
# =========================================
from __future__ import annotations

import math
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
from chart_utils import plot_candlestick_with_ma, PLOTLY_TV_CONFIG
from risk_grading import (
    grade_alpha,
    grade_sharpe,
    grade_treynor,
    grade_debt_equity,
    grade_current_ratio,
    grade_roe,
    summarize,
)

# ----------------- 格式工具 -----------------
def _fmt2(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.2f}"

def _fmt2pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x*100:.2f}%"

def _fmt0(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:,.0f}"

# ----------------- 市場資料 -----------------
@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    # 為風險指標提供市場報酬；優先台股指數，不行則用 S&P500
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

# ----------------- UI 區塊 -----------------
def _kpi(title: str, value: str, help_: str = ""):
    with st.container():
        st.metric(label=title, value=value, help=help_ or None)

def _kpi_grid(metrics: list[tuple[str, str, str]], cols: int = 4):
    if not metrics:
        return
    rows = (len(metrics) + cols - 1) // cols
    it = iter(metrics)
    for _ in range(rows):
        cs = st.columns(cols)
        for c in cs:
            try:
                name, val, hp = next(it)
            except StopIteration:
                break
            with c:
                _kpi(name, val, hp)

# ----------------- 主頁面 -----------------
def render(prefill_symbol: Optional[str] = None) -> None:
    st.header("股票")
    # 只保留日K，拿掉時間切換器，降低互動負擔
    col1 = st.columns([1])[0]
    with col1:
        default_kw = prefill_symbol or st.session_state.get("last_stock_kw", "2330")
        keyword = st.text_input("輸入股票代碼或名稱", value=default_kw)

    if not keyword:
        st.info("請輸入關鍵字（例：2330 或 台積電）")
        return

    try:
        ticker = find_ticker_by_name(keyword)       # 只回代碼
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_stock_kw"] = keyword

        # 嚴格分流：若是 ETF，直接提示並停止渲染
        if is_etf(ticker):
            st.warning("這是 ETF，請改至「ETF」分頁查詢。")
            return

        # ---- 準備 get_metrics 參數（近一年、rf=1%）----
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=365)
        market_close = _get_market_close_series(start, end)
        if market_close is None:
            st.error("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）")
            return
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=False)
        if not stats:
            st.error("查無此標的的歷史資料。")
            return

        # ======= 標題 =======
        with st.container(border=True):
            st.subheader(f"{name or ticker}（{ticker}）")

            # 風險等級 → 以 dict 傳入 summarize
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

            # 指標一覽（恢復並強化顯示）
            kpis = [
                ("Alpha", _fmt2(stats.get("Alpha")), "相對市場的超額報酬（年化）"),
                ("Beta", _fmt2(stats.get("Beta")), "相對市場波動敏感度"),
                ("Sharpe", _fmt2(stats.get("Sharpe Ratio")), "風險調整後報酬"),
                ("Treynor", _fmt2(stats.get("Treynor")), "以 Beta 計的風險調整報酬"),
                ("MADR", _fmt2(stats.get("MADR")), "最大向下偏離的均值幅度"),
                ("負債權益比", _fmt2(stats.get("負債權益比")), ""),
                ("流動比率", _fmt2(stats.get("流動比率")), ""),
                ("ROE", _fmt2pct(stats.get("ROE")), ""),
                ("股東權益", _fmt0(stats.get("Equity")), ""),
                ("EPS(TTM)", _fmt2(stats.get("EPS_TTM")), "近四季合計"),
            ]
            _kpi_grid(kpis, cols=5)

        # ======= 日 K 線 =======
        base_df: pd.DataFrame = stats["df"].copy()
        if not isinstance(base_df.index, pd.DatetimeIndex):
            base_df.index = pd.to_datetime(base_df.index)
        title = f"{name or ticker}（{ticker}）技術圖（日 K）"
        fig = plot_candlestick_with_ma(base_df, title=title, uirevision_key=f"{ticker}_D")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)

    except Exception as e:
        st.error(f"❌ 查詢股票失敗：{e}")

# 與 app.py 相容
def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
