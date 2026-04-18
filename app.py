import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
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
# Line Chart
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
# Compare Stocks
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
# LIVE DASHBOARD
# -------------------------------
st.subheader("⚡ Live Dashboard")

auto_refresh = st.checkbox("🔄 Auto Refresh (10 sec)", value=False)

placeholder = st.empty()

def run_dashboard():
    with placeholder.container():

        with st.spinner("Fetching live data..."):
            live_data = yf.download(ticker, period="1d", interval="1m")

        if live_data is None or live_data.empty:
            st.warning("Live data not available (market closed)")
            return

        latest = live_data.iloc[-1]

        # ✅ FIX 1: Safe current price
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

        # Safe features
        try:
            latest_features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
        except:
            st.warning("Incomplete live data")
            return

        # Prediction
        try:
            prediction = float(model.predict(latest_features)[0])
        except:
            st.warning("Prediction failed")
            return

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
        # ✅ FIX 2: Graph fallback
        # -------------------------------
        st.subheader("📉 Live Price Trend")

        if live_data['Close'].dropna().empty:
            st.warning("No live data — showing last 5 days")
            fallback = yf.download(ticker, period="5d")
            st.line_chart(fallback['Close'])
        else:
            st.line_chart(live_data['Close'])

        st.caption(f"Last updated: {latest.name}")

# Run once
run_dashboard()

# Auto refresh
if auto_refresh:
    time.sleep(10)
    st.experimental_rerun()
