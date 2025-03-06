# 
# Author: Luke Filicetti
# Description: 
# This program predicts the price of Bitcoin using a Long Short-Term Memory (LSTM) model. 
# It fetches historical data from the Binance API using the 'yfinance' library, processes the data, 
# and trains an LSTM model to predict future prices. The results are then plotted using matplotlib. 
# The model is capable of predicting the price for a few days into the future based on historical data.
#

# Import necessary libraries
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Fetch historical crypto prices from CoinGecko API
def get_historical_prices(days=60):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data from CoinGecko API: {response.status_code}")
        return None

    data = response.json()
    prices = [entry[1] for entry in data["prices"]]  # Extract closing prices
    dates = [datetime.now() - timedelta(days=i) for i in range(len(prices))]
    dates.reverse()  # Ensure chronological order

    return pd.DataFrame({"Date": dates, "Price": prices})

# Prepare data for LSTM training
def prepare_lstm_data(df, time_steps=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
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
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main script
df = get_historical_prices()
if df is not None:
    X_train, y_train, scaler = prepare_lstm_data(df)

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Predict future prices
    future_x = X_train[-1].reshape(1, X_train.shape[1], 1)
    future_price = model.predict(future_x)
    future_price = scaler.inverse_transform(future_price.reshape(-1, 1))

    print(f"Predicted next day's price: ${future_price[0][0]:.2f}")

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Price'], label="Historical Prices", color='blue')
    plt.scatter([df['Date'].max() + timedelta(days=1)], [future_price[0][0]], label="Predicted Price", color='red')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Prediction (LSTM)")
    plt.show()
