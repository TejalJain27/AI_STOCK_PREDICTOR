import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
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
    try:
        return yf.download(ticker, period="5y")
    except:
        return None

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
# Price Chart
# -------------------------------
st.subheader("📉 Stock Price Chart")

fig, ax = plt.subplots()
ax.plot(data['Close'], label="Closing Price")
ax.set_title("Stock Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# -------------------------------
# Compare Stocks (Simple)
# -------------------------------
st.subheader("📊 Compare Stocks")

compare_stocks = st.multiselect(
    "Select stocks to compare",
    list(stock_dict.keys()),
)

if compare_stocks:
    tickers = [stock_dict[s] for s in compare_stocks]
    comp_data = yf.download(tickers, period="1y")['Close']
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
# LIVE DASHBOARD
# -------------------------------
st.subheader("⚡ Live Dashboard")

def run_dashboard():
    with st.spinner("Fetching live data..."):

        live_data = yf.download(ticker, period="1d", interval="1m")

    if live_data is None or live_data.empty:
        st.warning("Live data not available (market closed)")
        return

    latest = live_data.iloc[-1]

    # Safe current price
    try:
        current_price = latest['Close']
        if hasattr(current_price, "values"):
            current_price = current_price.values[0]
        if current_price is None or np.isnan(current_price):
            st.warning("Live price not available yet")
            return
        current_price = float(current_price)
    except:
        st.warning("Error reading live price")
        return

    # Prediction
    try:
        latest_features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
        prediction = float(model.predict(latest_features)[0])
    except:
        st.warning("Prediction failed")
        return

    # Display
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

    # -------------------------------
    # Trend Graph (Final Stable)
    # -------------------------------
    st.subheader("📉 Live Price Trend")

    try:
        if hasattr(live_data.columns, "levels"):
            live_data.columns = [col[0] for col in live_data.columns]

        clean_data = live_data['Close'].dropna()

        if clean_data.empty:
            raise Exception("No live data")

        # Trend color
        if clean_data.iloc[-1] > clean_data.iloc[0]:
            color = "green"
            st.success("📈 Uptrend")
        else:
            color = "red"
            st.error("📉 Downtrend")

        fig2, ax2 = plt.subplots()
        ax2.plot(clean_data.index, clean_data.values, color=color)
        ax2.set_title("Live Price Trend")

        st.pyplot(fig2)

    except:
        st.warning("Showing last 5 days trend")

        fallback = yf.download(ticker, period="5d")
        fb = fallback['Close'].dropna()

        if not fb.empty:
            if fb.iloc[-1] > fb.iloc[0]:
                color = "green"
                st.success("📈 Uptrend (5 days)")
            else:
                color = "red"
                st.error("📉 Downtrend (5 days)")

            fig3, ax3 = plt.subplots()
            ax3.plot(fb.index, fb.values, color=color)
            st.pyplot(fig3)

# -------------------------------
# 🔥 Manual Refresh Button (FINAL FIX)
# -------------------------------
if st.button("🔄 Refresh Data"):
    run_dashboard()
