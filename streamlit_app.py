import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

# App Title
st.title("📈 SHIB/TUSD Forecasting Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload SHIB/TUSD CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

    # Column names
    df.columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df.drop(columns=['ignore'], inplace=True)

    st.subheader("Raw Data Preview")
    st.write(df.tail())

    # Close price plot
    st.subheader("🔹 Close Price Chart")
    st.line_chart(df['close'])

    # Volume plot
    st.subheader("🔹 Volume Chart")
    st.line_chart(df['volume'])

    # RSI Indicator
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['close'])
    st.subheader("🔹 RSI Indicator")
    st.line_chart(df['RSI'])

    # MACD Indicator
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    st.subheader("🔹 MACD & Signal")
    fig_macd, ax_macd = plt.subplots()
    ax_macd.plot(df.index, df['MACD'], label='MACD', color='green')
    ax_macd.plot(df.index, df['Signal'], label='Signal', color='red')
    ax_macd.legend()
    st.pyplot(fig_macd)

    # LSTM Forecast Section
    st.subheader("🔮 LSTM Price Forecast (Next 10 Steps)")

    # LSTM Modeling
    data = df[['close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length=60):
        X = []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
        return np.array(X)

    seq_len = 60
    X_all = create_sequences(scaled_data, seq_len)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_all[:-1], scaled_data[seq_len:-1], epochs=5, batch_size=32, verbose=0)

    # Future prediction
    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    future_preds = []

    current_input = last_60.copy()
    for _ in range(10):
        pred = model.predict(current_input)[0][0]
        future_preds.append(pred)
        next_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        current_input = next_input

    future_preds_original = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Plotting
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(df['close'][-100:].values, label='Recent Prices')
    ax_pred.plot(range(len(df)-1, len(df)+9), future_preds_original, label='Future Prediction', color='orange')
    ax_pred.set_title("LSTM Forecast")
    ax_pred.legend()
    st.pyplot(fig_pred)
