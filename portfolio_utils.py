import yfinance as yf

def evaluate_portfolio(portfolio):
    total_value = 0
    total_cost = 0
    detailed = []

    for stock in portfolio:
        try:
            ticker = stock['ticker']
            shares = stock['shares']
            cost = stock['cost']

            # 直接用現價抓取，避免 history 出錯
            info = yf.Ticker(ticker).fast_info
            price = info.get("lastPrice")

            if price is None:
                raise ValueError("無法取得現價")

            value = price * shares
            profit = (price - cost) * shares
            return_rate = ((price - cost) / cost) * 100

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