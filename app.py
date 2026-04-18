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
# Compare Stocks
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
# 🏆 Top Recommended Stock
# -------------------------------
st.subheader("🏆 Top Recommended Stock")

def get_best_stock():
    results = []

    for name, tick in stock_dict.items():
        try:
            df = yf.download(tick, period="1d", interval="1m")

            if df is None or df.empty:
                continue

            latest = df.iloc[-1]

            current = latest['Close']
            if hasattr(current, "values"):
                current = current.values[0]
            current = float(current)

            features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
            predicted = float(model.predict(features)[0])

            profit = predicted - current

            results.append((name, current, predicted, profit))

        except:
            continue

    if not results:
        return None

    best = sorted(results, key=lambda x: x[3], reverse=True)[0]

    if best[3] <= 0:
        return "NO_GOOD_STOCK"

    return best


if st.button("🔍 Find Best Stock"):
    best = get_best_stock()

    if best == "NO_GOOD_STOCK":
        st.warning("⚠ No stock is worth buying right now")

    elif best is None:
        st.warning("Could not fetch data")

    else:
        name, current, predicted, profit = best

        st.success(f"🏆 Best Stock: {name}")
        st.write(f"Current Price: ₹ {round(current,2)}")
        st.write(f"Predicted Price: ₹ {round(predicted,2)}")
        st.write(f"Expected Gain: ₹ {round(profit,2)}")
        st.success("📈 Recommended to BUY")

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

    try:
        current_price = latest['Close']
        if hasattr(current_price, "values"):
            current_price = current_price.values[0]
        current_price = float(current_price)
    except:
        st.warning("Error reading price")
        return

    try:
        features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
        prediction = float(model.predict(features)[0])
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
    # Trend Graph
    # -------------------------------
    st.subheader("📉 Live Price Trend")

    try:
        if hasattr(live_data.columns, "levels"):
            live_data.columns = [col[0] for col in live_data.columns]

        clean = live_data['Close'].dropna()

        if clean.empty:
            raise Exception

        color = "green" if clean.iloc[-1] > clean.iloc[0] else "red"

        if color == "green":
            st.success("📈 Uptrend")
        else:
            st.error("📉 Downtrend")

        fig2, ax2 = plt.subplots()
        ax2.plot(clean.index, clean.values, color=color)
        st.pyplot(fig2)

    except:
        st.warning("Showing last 5 days trend")

        fallback = yf.download(ticker, period="5d")
        fb = fallback['Close'].dropna()

        if not fb.empty:
            color = "green" if fb.iloc[-1] > fb.iloc[0] else "red"

            if color == "green":
                st.success("📈 Uptrend (5 days)")
            else:
                st.error("📉 Downtrend (5 days)")

            fig3, ax3 = plt.subplots()
            ax3.plot(fb.index, fb.values, color=color)
            st.pyplot(fig3)

# -------------------------------
# 🔥 AUTO LOAD + REFRESH
# -------------------------------
run_dashboard()  # Runs automatically on app load

if st.button("🔄 Refresh Data"):
    run_dashboard()
