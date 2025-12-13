import plotly.graph_objects as go

def plot_candlestick_with_ma(df, title=""):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], name="MA10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    return fig
