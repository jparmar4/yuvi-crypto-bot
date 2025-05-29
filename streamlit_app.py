import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

st.set_page_config(layout="wide")
st.title("📊 Crypto Signals Dashboard")

# Load the log file
log_file = "signal_log.csv"
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    st.warning("No signals found yet. Once your bot sends signals, they'll show up here.")
    st.stop()

# Sidebar for filters
st.sidebar.header("Filter Signals")
symbols = df['symbol'].unique()
intervals = df['interval'].unique()
selected_symbol = st.sidebar.selectbox("Select Coin", symbols)
selected_interval = st.sidebar.selectbox("Select Interval", intervals)

filtered = df[(df['symbol'] == selected_symbol) & (df['interval'] == selected_interval)]

# Show table of recent signals
st.subheader(f"Recent Signals for {selected_symbol} ({selected_interval})")
st.dataframe(filtered.sort_values("timestamp", ascending=False).head(25))

# Plot signal price chart
if not filtered.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["price"], mode="lines+markers", name="Price"))
    fig.add_trace(go.Scatter(x=filtered["timestamp"], y=filtered["rsi"], mode="lines", name="RSI", yaxis="y2"))
    # Mark buy/sell
    for idx, row in filtered.iterrows():
        if "LONG" in str(row["signal"]):
            fig.add_trace(go.Scatter(
                x=[row["timestamp"]], y=[row["price"]],
                mode="markers", marker=dict(color="green", size=10), name="LONG"
            ))
        elif "SHORT" in str(row["signal"]):
            fig.add_trace(go.Scatter(
                x=[row["timestamp"]], y=[row["price"]],
                mode="markers", marker=dict(color="red", size=10), name="SHORT"
            ))
    fig.update_layout(
        title=f"{selected_symbol} ({selected_interval}) - Signal Prices & RSI",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        yaxis2=dict(
            title="RSI",
            overlaying="y",
            side="right"
        ),
        height=600,
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No signals to plot for this coin/interval.")

# Show MACD/RSI values if you want
if st.checkbox("Show Full Signal Log"):
    st.write(filtered)

st.markdown(
    "Built with ❤️ using [Streamlit](https://streamlit.io/) - by your AI Trading Bot."
)
