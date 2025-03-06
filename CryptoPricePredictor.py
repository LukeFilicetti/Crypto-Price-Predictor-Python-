import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Fetch historical crypto prices (Replace with real API)
def get_historical_prices():
    days = 30
    prices = np.linspace(30000, 35000, days) + np.random.normal(0, 500, days)
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    return pd.DataFrame({'Date': dates, 'Price': prices})

# Predict future prices using Linear Regression
def predict_prices(data):
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Price']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array(range(data['Days'].max() + 1, data['Days'].max() + 11)).reshape(-1, 1)
    future_prices = model.predict(future_days)

    return future_days.flatten(), future_prices

# Main script
df = get_historical_prices()
future_x, future_y = predict_prices(df)

plt.figure(figsize=(10,5))
plt.scatter(df['Date'], df['Price'], label="Historical Prices", color='blue')
plt.plot([df['Date'].max() + timedelta(days=i) for i in range(1, 11)], future_y, label="Predicted Prices", color='red')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Bitcoin Price Prediction")
plt.show()