import streamlit as st
import yfinance as yf
import json
import os
from datetime import date
from stock_utils import TICKER_NAME_MAP

SAVE_PATH = "portfolio.json"

@st.cache_data(ttl=3600)
def get_latest_price(symbol: str):
    symbol = symbol.upper().strip()
    candidates = (
        [symbol] if symbol.endswith((".TW", ".TWO"))
        else [f"{symbol}.TW", f"{symbol}.TWO"]
    )
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

        col1, col2 = st.columns([6, 1])
        with col1:
            warn = " ⚠️" if stock["return"] < 0 else ""
            stock_name = TICKER_NAME_MAP.get(stock["ticker"], "")
            st.markdown(
                f"**{stock['ticker']}** {stock_name}｜"
                f"現價 {stock['price']}｜"
                f"股數 {stock['shares']}｜"
                f"市值 {stock['value']}｜"
                f"報酬率 {stock['return']}%{warn}"
            )
            st.caption(
                f"購買日：{stock['buy_date']}｜"
                f"買入金額：{stock['capital']} 元｜"
                f"未實現損益：{round(unrealized, 2)} 元"
            )

            if st.button("💰 售出", key=f"sell_btn_{idx}"):
                st.session_state[f"show_sell_{idx}"] = not st.session_state.get(f"show_sell_{idx}", False)

            if st.session_state.get(f"show_sell_{idx}", False):
                sell_qty = st.number_input("賣出股數", 1, stock["shares"], value=1, key=f"qty_{idx}")
                sell_price = st.number_input("賣出價格", min_value=0.0, step=0.01, format="%.2f", key=f"price_{idx}")
                sell_date = st.date_input("賣出日期", value=date.today(), key=f"date_{idx}")

                if st.button("🚀 確認售出", key=f"confirm_{idx}"):
                    proceeds = sell_qty * sell_price
                    cost_basis = sell_qty * stock["cost"]
                    realized = proceeds - cost_basis

                    st.session_state["pending_sale"] = {
                        "idx": idx,
                        "qty": sell_qty,
                        "price": sell_price,
                        "date": str(sell_date),
                        "proceeds": proceeds,
                        "realized": realized
                    }
                    st.rerun()

            if st.session_state.get("pending_sale") and st.session_state["pending_sale"]["idx"] == idx:
                ps = st.session_state["pending_sale"]
                st.warning(
                    f"⚠️ 即將售出 {stock['ticker']} 共 {ps['qty']} 股，售出價格 {ps['price']} 元，"
                    f"總計 {ps['proceeds']} 元（損益 {round(ps['realized'],2)} 元）"
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("⏪ 取消售出", key=f"cancel_{idx}"):
                        del st.session_state["pending_sale"]
                        st.rerun()
                with col_b:
                    if st.button("✅ 最終確認", key=f"final_{idx}"):
                        st.session_state["finalizing"] = ps
                        del st.session_state["pending_sale"]
                        st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.portfolio.pop(idx)
                save_portfolio()
                st.rerun()

    if "finalizing" in st.session_state:
        ps = st.session_state["finalizing"]
        with st.spinner("⏳ 正在執行售出，10 秒後完成…"):
            import time
            time.sleep(10)

        idx = ps["idx"]
        if idx < len(st.session_state.portfolio):
            stock = st.session_state.portfolio[idx]
            stock["shares"] -= ps["qty"]
            stock["capital"] = round(stock["shares"] * stock["cost"], 2)
            stock["value"] = round(stock["shares"] * stock["price"], 2)
            stock["return"] = round(((stock["value"] - stock["capital"]) / stock["capital"] * 100)
                                    if stock["capital"] > 0 else 0, 2)
            stock["realized_profit"] += round(ps["realized"], 2)
            if stock["shares"] == 0:
                st.session_state.portfolio.pop(idx)

        save_portfolio()
        del st.session_state["finalizing"]
        st.success("✅ 售出已完成！")
        st.rerun()

    st.divider()
    total_return = ((total_value - total_capital) / total_capital * 100) if total_capital > 0 else 0

    st.markdown(f"🔥 **總市值：{round(total_value,2)}**")
    st.markdown(f"💵 **總投入資金：{round(total_capital,2)}**")
    st.markdown(f"📉 **總報酬率：{round(total_return,2)}%**")
    st.caption(f"未實現損益：{round(total_unrealized,2)} 元")
    st.caption(f"🟩 已實現損益：{round(total_realized,2)} 元")
