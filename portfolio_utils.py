# /mnt/data/portfolio_utils.py
import yfinance as yf

def _normalize_tw_ticker(ticker: str):
    t = ticker.upper().strip()
    # 若已帶交易所後綴，直接回傳
    if t.endswith((".TW", ".TWO")):
        return [t]
    # 台股常見上市/上櫃嘗試
    return [f"{t}.TW", f"{t}.TWO"]

def evaluate_portfolio(portfolio):
    total_value = 0
    total_cost = 0
    detailed = []

    for stock in portfolio:
        try:
            ticker = stock['ticker']
            shares = stock['shares']
            cost = stock['cost']

            price = None
            # 與 portfolio_page.get_latest_price 行為對齊：嘗試兩種後綴
            for cand in _normalize_tw_ticker(ticker):
                try:
                    info = yf.Ticker(cand).fast_info
                    price = info.get("lastPrice")
                    if price:  # 取得即停
                        break
                except Exception:
                    continue

            if price is None:
                raise ValueError("無法取得現價")

            value = price * shares
            profit = (price - cost) * shares
            return_rate = ((price - cost) / cost) * 100 if cost != 0 else 0.0

            detailed.append({
                'ticker': ticker,
                'shares': shares,
                'cost': cost,
                'price': price,
                'value': value,
                'profit': profit,
                'return': return_rate,
            })

            total_value += value
            total_cost += cost * shares

        except Exception as e:
            detailed.append({
                'ticker': stock.get('ticker', '未知'),
                'error': f"取得資料失敗：{e}"
            })

    total_return = ((total_value - total_cost) / total_cost * 100) if total_cost != 0 else 0

    return {
        'total_value': total_value,
        'total_return': total_return,
        'detailed': detailed
    }
