import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime
from pandas import ExcelWriter
import xlsxwriter
import os, sys, subprocess
plt.style.use('fivethirtyeight')
'''df = pd.read_csv('/Users/lakshaymittal/Downloads/500209.csv', header=0,
                      index_col='Date',
                      parse_dates=True).iloc[::-1]

#Create a new dataframe with only the 'Close' column
data = df.filter(['Close Price'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)

#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

    # Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


#Convert x_test to a numpy array
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

print(valid)
last_60_days = data[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR ', fontsize=18)
plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()'''
df = pd.read_csv('/Users/lakshaymittal/Downloads/500209.csv', header=0,
                     index_col='Date',
                     parse_dates=True).iloc[::-1]

 # Create a new dataframe with only the 'Close' column
data = df.filter(['Close Price'])
    # Converting the dataframe to a numpy array
dataset = data.values
    # Get /Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

    # Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

    # Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

        # Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

    # Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Test data set
test_data = scaled_data[training_data_len - 60:, :]
    # Create the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len:,
             :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to a numpy array
x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

    # Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

print(valid)
last_60_days = data[-60:].values
    # Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
    # Create an empty list
X_test = []
    # Append teh past 60 days
X_test.append(last_60_days_scaled)
    # Convert the X_test data set to a numpy array
X_test = np.array(X_test)
    # Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Get the predicted scaled price
pred_price = model.predict(X_test)
    # undo the scaling
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

with ExcelWriter('/Users/lakshaymittal/PycharmProjects/ML_robot/Forecast_Excel/500209.xlsx',
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

open_file('/Users/lakshaymittal/PycharmProjects/ML_robot/Forecast_Excel/500209.xlsx')


