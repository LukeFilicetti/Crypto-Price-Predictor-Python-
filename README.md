# Crypto-Price-Predictor-Python-
Description: Uses historical crypto price data (e.g., Bitcoin, Ethereum) to predict future prices using time series forecasting.

Fetches live cryptocurrency prices from CoinGecko API.
Uses Long Short-Term Memory (Recurrent Neural Network) for forecasting.
Visualizes price trends and predicts future values.

LSTMs are a type of neural network architecture well-suited for handling sequential data, which makes them a great choice for time series data like stock or cryptocurrency prices.

# Crypto Price Predictor Setup and Usage Guide
Setting Up the Environment
1. Clone the Repository:

git clone https://github.com/<your-username>/Crypto-Price-Predictor-Python.git

2. Create and Activate the Virtual Environment:

cd Crypto-Price-Predictor-Python

For Windows:

python -m venv venv
venv\Scripts\activate


3. Installing Required Libraries:

pip install tensorflow pandas numpy requests matplotlib scikit-learn yfinance

4. Running the Script:

python CryptoPricePredictor.py

5. Deactivating the Virtual Environment:

deactivate

6. Switching Between Cryptos:
Currently, the script is set to track Bitcoin (BTC). To track a different cryptocurrency, modify the following line in the CryptoPricePredictor.py:

symbol = 'BTCUSDT'

Change 'BTCUSDT' to the symbol of the cryptocurrency you want to track. For example:

Ethereum (ETH): 'ETHUSDT'
Litecoin (LTC): 'LTCUSDT'
XRP: 'XRPUSDT'

7. Available Cryptocurrencies
You can find the symbols of various cryptocurrencies from the CoinGecko API. Here are a few examples:

Bitcoin: 'BTCUSDT'
Ethereum: 'ETHUSDT'
Litecoin: 'LTCUSDT'
XRP: 'XRPUSDT'

8. Troubleshooting:
If you encounter any issues with the TensorFlow or Keras imports (due to Pylance warnings), don’t worry as long as the script runs correctly. These warnings are often related to the VS Code environment settings.
If the script takes too long or encounters an issue during training, try reducing the dataset size by modifying the number of days of historical data in the function:

get_historical_prices(symbol='BTCUSDT', days=60)