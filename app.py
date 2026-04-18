import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 Real-Time AI Stock Predictor")

# -------------------------------
# Stock Selection
# -------------------------------
stock_dict = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

selected_stock = st.selectbox("Select Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, period="5y")

data = load_data(ticker)

if data is None or data.empty:
    st.error("No data found.")
    st.stop()

# -------------------------------
# Historical Data
# -------------------------------
st.subheader("📊 Historical Data")
st.dataframe(data.tail())

# -------------------------------
# Candlestick Chart
# -------------------------------
st.subheader("🕯️ Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Multi-Stock Comparison (Feature)
# -------------------------------
st.subheader("📊 Compare Stocks")

compare = st.checkbox("Compare with other stocks")

if compare:
    compare_stocks = st.multiselect(
        "Select stocks to compare",
        list(stock_dict.keys()),
        default=["TCS"]
    )

    comp_data = yf.download([stock_dict[s] for s in compare_stocks], period="1y")['Close']

    st.line_chart(comp_data)

# -------------------------------
# ML Model
# -------------------------------
data['Prediction'] = data['Close'].shift(-1)
data = data.dropna()

X = data[['Open','High','Low','Close','Volume']]
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.subheader("🤖 Model Accuracy")
st.write(round(accuracy, 4))

# -------------------------------
# ⚡ LIVE DASHBOARD (SAFE VERSION)
# -------------------------------
st.subheader("⚡ Live Dashboard")

auto_refresh = st.checkbox("🔄 Auto Refresh (10 sec)", value=False)

placeholder = st.empty()

def run_dashboard():
    with placeholder.container():

        with st.spinner("Fetching live data..."):
            live_data = yf.download(ticker, period="1d", interval="1m")

        if live_data is None or live_data.empty:
            st.error("No live data available")
            return

        latest = live_data.iloc[-1]

        if latest[['Open','High','Low','Close','Volume']].isnull().any():
            st.warning("Incomplete data, retrying...")
            return

        latest_features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)

        current_price = float(latest['Close'])
        prediction = float(model.predict(latest_features)[0])

        col1, col2, col3 = st.columns(3)

        col1.metric("Current Price", f"₹ {round(current_price,2)}")
        col2.metric("Predicted Price", f"₹ {round(prediction,2)}")

        delta = prediction - current_price

        if delta > 0:
            col3.metric("Signal", "BUY 📈", f"+{round(delta,2)}")
            st.success("Strong Buy Signal 🚀")
        else:
            col3.metric("Signal", "SELL 📉", f"{round(delta,2)}")
            st.error("Sell Signal ⚠")

        st.subheader("📉 Live Price Trend")
        st.line_chart(live_data['Close'])

        st.caption(f"Last updated: {latest.name}")

# Run once
run_dashboard()

# Auto refresh safely
if auto_refresh:
    time.sleep(10)
    st.experimental_rerun()
