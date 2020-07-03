import matplotlib.pyplot as plt
from  matplotlib import style
import datetime
import openpyxl
from pandas import ExcelWriter
import xlsxwriter
import os, sys, subprocess
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
valid= pd.DataFrame()
train = pd.DataFrame()
pred_price=0


def regFunc(sec_no1):
    sec_no = str(sec_no1)
    global valid,train,pred_price


    df = pd.read_csv('/Downloads/'+sec_no+'.csv', header=0,
                     index_col='Date',
                     parse_dates=True).iloc[::-1]


    data = df.filter(['Close Price'])

    dataset = data.values

    training_data_len = math.ceil(len(dataset) * .8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:,
             :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])


    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    rmse

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    print(valid)
    last_60_days = data[-60:].values

    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []

    X_test.append(last_60_days_scaled)

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)

    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)


    last_date = valid.iloc[-1].name
    excel_date = valid.iloc[-60].name
    excel_unix = excel_date.timestamp()
    last_unix = last_date.timestamp()
    one_day = 86400
    for_excel = datetime.datetime.fromtimestamp(last_unix + one_day)
    next_unix = last_unix + one_day


    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    valid.loc[next_date] = [np.nan for _ in range(len(valid.columns) - 1)] + [pred_price]

    excel_date = valid.iloc[-60].name
    excel_unix = excel_date.timestamp()
    for_excel = datetime.datetime.fromtimestamp(excel_unix + one_day)

    with ExcelWriter('/Stock-Prediction-Using-CNN/Forecast_Excel/' + sec_no + '.xlsx',
                     datetime_format='mmm d yyyy hh:mm:ss',
                     date_format='mmmm dd yyyy', engine='xlsxwriter') as writer:
        valid[['Close Price','Predictions']].loc[for_excel:].to_excel(writer)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': 'dd/mm/yy hh:mm:ss'})
        worksheet.set_column('A:A', 18, format1)
        worksheet.set_column('B:B', 14)
        writer.save()

    def open_file(filename):
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    open_file('/Stock-Prediction-Using-CNN/Forecast_Excel/' + sec_no + '.xlsx')
    return
