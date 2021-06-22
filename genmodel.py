import os
import math
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as web

from data import RealtimeData
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


class ModelGen:
    def __init__(self, symbol, start, end, base_days):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.base_days = base_days

    def get_data(self, train_split):
        df = RealtimeData(self.symbol).get_historical(self.start, self.end)
        data = df.filter(['close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = math.ceil(len(scaled_data) * train_split)
        train_data = scaled_data[:train_size, :]
        x_train = []
        y_train = []

        for i in range(self.base_days, len(train_data)):
            x_train.append(train_data[i - self.base_days:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_data[train_size - self.base_days:, :]
        x_test = []
        y_test = data[train_size:, :]

        for i in range(self.base_days, len(test_data)):
            x_test.append(test_data[i - self.base_days:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        print(x_test.shape, y_test.shape)

        return data, scaler, x_train, y_train, x_test, y_test

    def create_model(self):
        train_split = 0.8
        data, scaler, x_train, y_train, x_test, y_test = self.get_data(train_split)
        model = build_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=16, epochs=50)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        print('Test result: {}'.format(rmse))

        os.mkdir('models/{}'.format(self.symbol))
        model.save('models/{}/model.h5'.format(self.symbol))
        pickle.dump(scaler, open('models/{}/scaler.pkl'.format(self.symbol), 'wb'))


today = dt.date.today()
model_generator = ModelGen('AAPL', '2018-01-01', today, 60)
model_generator.create_model()

model_generator = ModelGen('TSLA', '2018-01-01', today, 60)
model_generator.create_model()

model_generator = ModelGen('AMZN', '2018-01-01', today, 60)
model_generator.create_model()
