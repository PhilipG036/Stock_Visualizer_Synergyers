import numpy as np
import pandas as pd
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

def load_ds():
    return pd.read_csv('./data/stockNames.csv', usecols=['Symbol', 'Name'])

def load_stock(user_input: str):
    stock_info = yf.download(user_input, start='2023-01-01', end=f'{date.today()}', progress=False)
    stock_info.drop('Volume', axis = 1, inplace = True)
    stock_info.index = pd.to_datetime(stock_info.index).date
    stock_info.sort_index(inplace=True, ascending=False)
    return stock_info

def data_preprocess(forecast_days: int, user_input: str):
    stock_info = load_stock(user_input)
    stock_prediction = stock_info[['Adj Close']]
    stock_prediction['Stock Price'] = stock_prediction.loc[:, 'Adj Close'].shift(-forecast_days)
    return stock_prediction

def data_prep(forecast_days: int, user_input: str):
    stock = data_preprocess(forecast_days, user_input)
    # CREATE X DATASET
    X_DATA = np.array(stock.drop('Stock Price', axis=1))
    X_DATA = X_DATA[:-forecast_days]
    # CREATE Y DATASET
    Y_DATA = np.array(stock['Stock Price'])
    Y_DATA = Y_DATA[:-forecast_days]
    # TEST SPLIT TRAIN DATA
    x_train, x_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size = 0.2)
    return x_train, x_test, y_train, y_test

def model(forecast_days: int, user_input: str, m: object):
    stock = data_preprocess(forecast_days, user_input)
    x_train, x_test, y_train, y_test = data_prep(forecast_days, user_input)
    m_fit = m.fit(x_train, y_train)
    stock_price_pred = np.array(stock.drop(columns='Stock Price', axis=1))[forecast_days:]
    pred = m_fit.predict(stock_price_pred) # [[c1, c2.], []]
    return pred[:forecast_days]


# ml.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def lstm_model(X: list):
    X_DATA=X
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_DATA.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def data_preprocess_lstm(forecast_days: int, user_input: str):
    stock_info = load_stock(user_input)
    stock_info = stock_info['Adj Close'].values.reshape(-1, 1)
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_info_scaled = scaler.fit_transform(stock_info)
    # Create input data
    X, Y = [], []
    for i in range(forecast_days, len(stock_info_scaled)):
        X.append(stock_info_scaled[i - forecast_days:i, 0])
        Y.append(stock_info_scaled[i, 0])
    X, Y = np.array(X), np.array(Y)
    # Reshape the data for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y, scaler

def model_lstm(forecast_days: int, user_input: str):
    X, Y, scaler = data_preprocess_lstm(forecast_days, user_input)
    model = lstm_model(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    model.fit(x_train, y_train, epochs=50, batch_size=32)
    stock_price_pred = X[-forecast_days:]
    pred_scaled = model.predict(stock_price_pred)
    # Inverse transform to get the actual predicted values
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred


from statsmodels.tsa.arima.model import ARIMA

def model_arima(forecast_days: int, user_input: str):
    stock_info = data_preprocess(forecast_days, user_input)
    # Convert the date index to datetime
    stock_info.index = pd.to_datetime(stock_info.index)
    # Create ARIMA model
    model = ARIMA(stock_info['Adj Close'], order=(5, 1, 0))  # Adjust order as needed
    # Fit the model
    model_fit = model.fit()
    # Forecast future values
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

algo = {'svm': SVR(kernel='rbf', C = 1000.0, gamma = 0.0001), 'tree': RandomForestRegressor(), 'lstm':model_lstm, 'arima':model_arima}