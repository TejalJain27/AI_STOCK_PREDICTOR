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
    data = yf.download(ticker, period="5y", progress=False)
    if data is None or data.empty:
        data = yf.download(ticker, period="1y", progress=False)
    return data

data = load_data(ticker)

if data is None or data.empty:
    st.error("❌ Data not available")
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
ax.legend()
st.pyplot(fig)

# -------------------------------
# 📊 Compare Stocks (FIXED)
# -------------------------------
st.subheader("📊 Compare Stocks")

compare_stocks = st.multiselect(
    "Select stocks to compare",
    list(stock_dict.keys())
)

if compare_stocks:
    tickers = [stock_dict[s] for s in compare_stocks]
    comp_data = yf.download(tickers, period="1y", progress=False)

    if comp_data is None or comp_data.empty:
        st.warning("⚠ Comparison data not available")
    else:
        try:
            comp_close = comp_data['Close']

            if len(compare_stocks) == 1:
                comp_close = comp_close.to_frame(name=compare_stocks[0])

            st.line_chart(comp_close)

        except:
            st.warning("⚠ Error displaying comparison")

# -------------------------------
# ML Model
# -------------------------------
data['Prediction'] = data['Close'].shift(-1)
data = data.dropna()

X = data[['Open','High','Low','Close','Volume']]
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.subheader("🤖 Model Accuracy")
st.write(round(accuracy, 4))

# -------------------------------
# 🏆 Top Recommended Stock (FIXED)
# -------------------------------
st.subheader("🏆 Top Recommended Stock")

def get_best_stock():
    results = []

    for name, tick in stock_dict.items():
        try:
            df = yf.download(tick, period="5d", interval="1d", progress=False)

            if df is None or df.empty:
                continue

            latest = df.iloc[-1]

            current = latest['Close']
            if hasattr(current, "values"):
                current = current.values[0]
            current = float(current)

            features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
            prediction = model.predict(features)
            prediction = float(prediction[0])

            profit = prediction - current
            results.append((name, current, prediction, profit))

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

# -------------------------------
# ⚡ Live Dashboard (FIXED)
# -------------------------------
st.subheader("⚡ Live Dashboard")

def run_dashboard():
    live_data = yf.download(ticker, period="5d", interval="1d", progress=False)

    if live_data is None or live_data.empty:
        st.warning("Live data not available")
        return

    latest = live_data.iloc[-1]

    # FIX: Safe float conversion
    current_price = latest['Close']
    if hasattr(current_price, "values"):
        current_price = current_price.values[0]
    current_price = float(current_price)

    features = latest[['Open','High','Low','Close','Volume']].values.reshape(1,-1)
    prediction = model.predict(features)
    prediction = float(prediction[0])

    st.write("### Current Price:", round(current_price, 2))
    st.write("### Predicted Price:", round(prediction, 2))

    if prediction > current_price:
        st.success("📈 BUY Signal")
    else:
        st.error("📉 SELL Signal")

    # -------------------------------
    # 📉 Trend (FIXED)
    # -------------------------------
    st.subheader("📉 Trend")

    close = live_data['Close'].dropna()

    # FIX: Convert to scalar safely
    last = close.iloc[-1]
    first = close.iloc[0]

    if hasattr(last, "values"):
        last = last.values[0]
    if hasattr(first, "values"):
        first = first.values[0]

    color = "green" if last > first else "red"

    values = close.values
    if len(values.shape) > 1:
        values = values.flatten()

    fig2, ax2 = plt.subplots()
    ax2.plot(close.index, values, color=color)
    st.pyplot(fig2)

# -------------------------------
# AUTO LOAD + REFRESH
# -------------------------------
run_dashboard()

if st.button("🔄 Refresh"):
    run_dashboard()
