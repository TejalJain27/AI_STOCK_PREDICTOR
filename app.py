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
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    return data

data = load_data(ticker)

if data.empty:
    st.error("No data found.")
    st.stop()

# -------------------------------
# Show Data
# -------------------------------
st.subheader("📊 Historical Data")
st.dataframe(data.tail())

# -------------------------------
# Plot Graph
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
# Prepare ML Model
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
# Live Prediction
# -------------------------------
st.subheader("⚡ Live Prediction")

if st.button("Get Live Prediction"):

    live_data = yf.download(ticker, period="1d", interval="1m")

    if live_data.empty:
        st.error("No live data available")
    else:
        latest = live_data.iloc[-1]

        latest_features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)

        prediction = model.predict(latest_features)[0]
        current_price = float(latest['Close'])

        st.write("### Current Price:", round(current_price, 2))
        st.write("### Predicted Next Price:", round(prediction, 2))

        # Buy/Sell Signal
        if prediction > current_price:
            st.success("📈 BUY Signal")
        else:
            st.error("📉 SELL Signal")