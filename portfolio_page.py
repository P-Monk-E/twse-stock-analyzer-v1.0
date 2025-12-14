# ...（保留原本 import 區）

import streamlit as st
import yfinance as yf
import json, os
from datetime import date
from urllib.parse import quote

SAVE_PATH = "portfolio.json"

@st.cache_data(ttl=3600)
def get_latest_price(symbol: str):
    symbol = symbol.upper().strip()
    candidates = [symbol] if symbol.endswith((".TW", ".TWO")) else [f"{symbol}.TW", f"{symbol}.TWO"]
    for tkr in candidates:
        try:
            hist = yf.Ticker(tkr).history(period="5d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return None

def save_portfolio():
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state.portfolio, f, ensure_ascii=False, indent=2)

def load_portfolio():
    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                st.session_state.portfolio = json.load(f)
        except Exception:
            st.session_state.portfolio = []

def is_etf(ticker):
    return ticker.startswith("00") or ticker.startswith("ETF") or ticker in {"0050", "0056", "006208"}

def show():
    st.header("📦 庫存")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []
        load_portfolio()

    st.subheader("加入 股票 / ETF")
    ticker = st.text_input("加入 股票 / ETF（代碼或名稱）").strip().upper()
    shares = st.number_input("股數", min_value=1, step=1, value=1)
    cost = st.number_input("成本價", min_value=0.0, step=0.01, format="%.2f", value=0.0)
    buy_date = st.date_input("購買日期", value=date.today())

    if st.button("加入"):
        if not ticker:
            st.warning("⚠️ 請輸入股票 / ETF 代碼")
        else:
            price = get_latest_price(ticker)
            if price is None:
                st.error("❌ 無法取得該股票 / ETF 的最新價格")
            else:
                capital = cost * shares
                value = price * shares
                rtn = ((value - capital) / capital) * 100 if capital > 0 else 0

                st.session_state.portfolio.append({
                    "ticker": ticker,
                    "shares": shares,
                    "cost": round(cost, 2),
                    "price": round(price, 2),
                    "capital": round(capital, 2),
                    "value": round(value, 2),
                    "return": round(rtn, 2),
                    "buy_date": buy_date.strftime("%Y-%m-%d"),
                    "realized_profit": 0.0
                })
                save_portfolio()
                st.success(f"✅ {ticker} 已加入庫存（現價 {round(price,2)}）")
                st.rerun()

    st.divider()
    st.subheader("📊 持股清單")

    if not st.session_state.portfolio:
        st.info("目前尚無持股")
        return

    for idx, stock in enumerate(st.session_state.portfolio):
        col1, col2 = st.columns([6, 1])
        with col1:
            ticker = stock["ticker"]
            link_target = "ETF" if is_etf(ticker) else "股票"
            link = f"?page={quote(link_target)}&symbol={quote(ticker)}"
            st.markdown(
                f"[**{ticker}**]({link})｜現價 {stock['price']}｜市值 {stock['value']}｜報酬率 {stock['return']}%"
            )
            st.caption(f"購買日：{stock['buy_date']}｜持有股數：{stock['shares']}｜未實現損益：{round(stock['value'] - stock['capital'], 2)} 元")

#（此處省略後續售出邏輯，與之前一致）
