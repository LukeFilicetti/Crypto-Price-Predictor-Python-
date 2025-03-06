import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Fetch historical crypto prices from Binance API
def get_historical_prices(symbol="BTCUSDT", days=60):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1d", "limit": days}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data from Binance API: {response.status_code}")
        return None

    data = response.json()
    prices = [float(entry[4]) for entry in data]  # Closing prices
    dates = [datetime.now() - timedelta(days=i) for i in range(len(prices))]
    dates.reverse()  # Ensure chronological order

    return pd.DataFrame({'Date': dates, 'Price': prices})

# Prepare data for LSTM training
def prepare_lstm_data(df, time_steps=10):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Price']])

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    return X, y, scaler

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model every time before making predictions
def train_and_predict(symbol="BTCUSDT", time_steps=10, future_days=10):
    # Fetch latest prices
    df = get_historical_prices(symbol)
    if df is None:
        return
    
    # Prepare data
    X_train, y_train, scaler = prepare_lstm_data(df, time_steps)
    
    # Build and train model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Predict future prices
    last_sequence = df['Price'].values[-time_steps:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    future_predictions = []
    for _ in range(future_days):
        X_input = np.array([last_sequence_scaled])
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        
        prediction = model.predict(X_input, verbose=0)[0, 0]
        future_predictions.append(prediction)

        last_sequence_scaled = np.append(last_sequence_scaled[1:], prediction).reshape(-1, 1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(df['Date'], df['Price'], label="Historical Prices", color='blue')
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, 11)]
    plt.plot(future_dates, future_prices, label="Predicted Prices (LSTM)", color='red')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{symbol} Price Prediction (Binance + LSTM)")
    plt.show()

# Run prediction
train_and_predict("BTCUSDT")  # Change to "ETHUSDT", "DOGEUSDT", etc.