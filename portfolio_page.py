import streamlit as st
import yfinance as yf
import json
import os
from datetime import date

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
                    "cost": round(cost,2),
                    "price": round(price,2),
                    "capital": round(capital,2),
                    "value": round(value,2),
                    "return": round(rtn,2),
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

    total_value = 0
    total_capital = 0
    total_unrealized = 0
    total_realized = 0

    for idx, stock in enumerate(st.session_state.portfolio):
        total_value += stock["value"]
        total_capital += stock["capital"]
        unrealized = stock["value"] - stock["capital"]
        total_unrealized += unrealized
        total_realized += stock.get("realized_profit", 0.0)

        col1, col2 = st.columns([6,1])
        with col1:
            warn = " ⚠️" if stock["return"] < 0 else ""
            target_page = "股票" if stock["ticker"].isdigit() else "ETF"
            st.markdown(
                f"**[{stock['ticker']}（{stock['shares']}股）](/?page={target_page}&symbol={stock['ticker']})**｜成本 {stock['cost']}｜現價 {stock['price']}｜市值 {stock['value']}｜報酬率 {stock['return']}%{warn}"
            )
            st.caption(f"購買日：{stock['buy_date']}｜未實現損益：{round(unrealized, 2)} 元")

        with col2:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.portfolio.pop(idx)
                save_portfolio()
                st.rerun()

    st.divider()
    total_return = ((total_value - total_capital) / total_capital * 100) if total_capital > 0 else 0
    st.markdown(f"🔥 **總市值：{round(total_value,2)}**")
    st.markdown(f"💵 **總投入資金：{round(total_capital,2)}**")
    st.markdown(f"📉 **總報酬率：{round(total_return,2)}%**")
    st.caption(f"未實現損益：{round(total_unrealized,2)} 元")
    st.caption(f"🟩 已實現損益：{round(total_realized,2)} 元")
