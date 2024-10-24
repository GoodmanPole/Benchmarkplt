# Import Libraries
import datetime
import math
import os


from predictmodel.performance_metric import PerformanceMetrics

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# Holt Winters
# Single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# Double and Triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima.utils import ndiffs




class PredictionModels:
    def __init__(self):
        pass

    # def plotter(self,rawDataset,trainDatset,predictedDatset):
    #     # Visualize the data
    #     plt.figure(figsize=(16, 8))
    #     plt.title(f'Model Result for {f}')
    #     plt.xlabel('Date', fontsize=18)
    #     plt.ylabel('Close Price USD ($)', fontsize=18)
    #     plt.plot(train['Close'])
    #     plt.plot(valid[['Close', 'Predictions']])
    #     plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
    #     plt.show()


    def str_to_datetime(self,s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)


    def df_to_windowed_df(self,dataframe, first_date_str, last_date_str, n=3):
        print(dataframe)
        first_date = self.str_to_datetime(first_date_str)
        last_date = self.str_to_datetime(last_date_str)

        target_date = first_date

        dates = []
        X, Y = [], []

        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n + 1)
            # print(df_subset)

            if len(df_subset) != n + 1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

            if last_time:
                break

            target_date = next_date

            if target_date == last_date:
                last_time = True

        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates

        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n - i}'] = X[:, i]

        ret_df['Target'] = Y

        return ret_df

    def windowed_df_to_date_X_y(self,windowed_dataframe):
        df_as_np = windowed_dataframe.to_numpy()

        dates = df_as_np[:, 0]

        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        Y = df_as_np[:, -1]

        return dates, X.astype(np.float64), Y.astype(np.float64)

    def lstm(self):

        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                plt.figure(figsize=(16, 8))
                plt.title(f'Closing Price History {f}')
                plt.plot(cryptoData['Close'])
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Get /Compute the number of rows to train the model on
                training_data_close_price_len = math.ceil(len(closing_price_dataset) * .8)
                print(training_data_close_price_len)

                # Scale the all of the data to be values between 0 and 1
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_closing_price_data = scaler.fit_transform(closing_price_dataset)
                print(scaled_closing_price_data)
                print()
                print(0, len(scaled_closing_price_data))

                # Create the scaled training data set
                train_closing_price_data = scaled_closing_price_data[0:training_data_close_price_len, :]
                # Split the data into x_train and y_train data sets
                x_train = []
                y_train = []
                for i in range(60, len(train_closing_price_data)):
                    x_train.append(train_closing_price_data[i - 60:i, 0])
                    y_train.append(train_closing_price_data[i, 0])
                    # if i<=61:
                    #   print(x_train)
                    #   print(y_train)
                    #   print()

                # Convert x_train and y_train to numpy arrays
                x_train, y_train = np.array(x_train), np.array(y_train)
                # print(y_train)
                # print(y_train.shape)

                # Reshape the data into the shape accepted by the LSTM because it is 2D and we must reshape it 3D
                # We input the (number_of_samples,number,of,timestamps,number_of_features)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))

                # #Build the LSTM network model
                model = Sequential()
                # input_shape contains number of timestamps and number of features
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                # print("ok")

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, batch_size=1, epochs=1)

                # Test data set
                test_closing_price_data = scaled_closing_price_data[training_data_close_price_len - 60:, :]
                print(1, len(test_closing_price_data))

                # Create the x_test and y_test data sets
                x_test = []
                y_test = closing_price_dataset[training_data_close_price_len:,
                         :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2032 - 1972 = 60 rows of data
                for i in range(60, len(test_closing_price_data)):
                    x_test.append(test_closing_price_data[i - 60:i, 0])

                # Convert x_test to a numpy array
                x_test = np.array(x_test)
                print(2, len(x_test))

                # Reshape the data into the shape accepted by the LSTM
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Getting the models predicted price values
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)  # Undo scaling

                # Calculate/Get the value of RMSE
                rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

                # Plot/Create the data for the graph
                train = data[:training_data_close_price_len]
                valid = data[training_data_close_price_len:]
                valid['Predictions'] = predictions
                # Visualize the data
                plt.figure(figsize=(16, 8))
                plt.title(f'Model Result for {f}')
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.plot(train['Close'])
                plt.plot(valid[['Close', 'Predictions']])
                plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
                plt.show()

                print(rmse)

    def arima(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                plt.figure(figsize=(16, 8))
                plt.title(f'Closing Price History {f}')
                plt.plot(cryptoData['Close'])
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:
                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = int(ndiffs(closing_price_dataset_log, test="adf"))

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = len(test_dataset)
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(10, 8))
                    plt.plot(np.exp(train_dataset), label='Train')
                    plt.plot(np.exp(test_dataset), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = int(ndiffs(closing_price_dataset_log, test="adf"))

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

    def ar(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                # plt.figure(figsize=(16, 8))
                # plt.title(f'Closing Price History {f}')
                # plt.plot(cryptoData['Close'])
                # plt.xlabel('Date', fontsize=18)
                # plt.ylabel('Close Price USD ($)', fontsize=18)
                # plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:

                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = 0

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = 0

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

    def ma(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                # plt.figure(figsize=(16, 8))
                # plt.title(f'Closing Price History {f}')
                # plt.plot(cryptoData['Close'])
                # plt.xlabel('Date', fontsize=18)
                # plt.ylabel('Close Price USD ($)', fontsize=18)
                # plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:

                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = 0

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = 0

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()


    def holt_winters_single_exp(self):

        # open datasets file
        for f in os.listdir('D:\\pycharm\\Goodman_Predict_Software/Dataset'):
            print(f)
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'D:\\pycharm\\Goodman_Predict_Software/Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.iloc[::-1]
                cryptoData = cryptoData.drop('Currency', axis=1)
                print(cryptoData)

                # Making the Date column readable
                cryptoData = cryptoData.reset_index()
                cryptoData['Date'] = pd.to_datetime(cryptoData['Date']).dt.date
                cryptoData['Date'] = pd.to_datetime(cryptoData['Date'], format='%Y-%m-%d')
                cryptoData=cryptoData[['Date','Close']]
                cryptoData.index = cryptoData.pop('Date')
                print(cryptoData)

                cryptoData=self.df_to_windowed_df(cryptoData,'2017-07-28','2021-01-01',n=3)

                print(cryptoData)
                dates, X, y = self.windowed_df_to_date_X_y(cryptoData)
                q_80 = int(len(dates) * .8)
                dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
                dates_test, X_test, y_test = dates[q_80:], X[q_80:], y[q_80:]
                # print(cryptoData.shape)
                # print(cryptoData.head())

                # plotting the original data
                ''' 
                Here we Look for:
                Level: The average value in the series.
                Trend: The increasing or decreasing value in the series.
                Seasonality: The repeating short-term cycle in the series.
                Noise: The random variation in the series.
                '''

                # cryptoData[['Close']].plot(title='Close Price')
                # decompose_result = seasonal_decompose(cryptoData['Close'], model ='multiplicative')
                # decompose_result.plot()

                # Set the value of Alpha and define m (Time Period)
                m = 365
                alpha = 1 / (2 * m)

                # Splitting the dataset to train and test
                # train_df= cryptoData[:int(0.8*cryptoData['Close'].shape[0])]
                # test_df=cryptoData[int(0.8*cryptoData['Close'].shape[0]):]




                fitted_model = SimpleExpSmoothing(X_train['Close']).fit(smoothing_level=alpha, optimized=False, use_brute=True)
                prediction_df= fitted_model.forecast(X_test.shape[0])
                # prediction.index=prediction.index.astype('str')


                # print(type(prediction.index))
                X_train['Close'].plot(legend=True,label='Train')
                X_test['Close'].plot(legend=True,label='Test',figsize=(9,4))
                prediction_df.plot(legend=True,label='Prediction')
                # cryptoData[['Close', 'HWS1']].plot(title='Holt Winters Single Exponential Smoothing')
                plt.title('Holt Winter Single Exp Smoothing')

                plt.show()
                performance = PerformanceMetrics()
                mse = performance.mse(X_test['Close'], prediction_df)
                print("Single MSE")
                print(mse)




    def holt_winters_double_exp(self):

        # open datasets file
        for f in os.listdir('D:\\pycharm\\Goodman_Predict_Software/Dataset'):
            print(f)
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'D:\\pycharm\\Goodman_Predict_Software/Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData=cryptoData.iloc[::-1]
                cryptoData = cryptoData.drop('Currency', axis=1)

                # print(cryptoData.shape)
                # print(cryptoData.head())

                # plotting the original data
                ''' 
                Here we Look for:
                Level: The average value in the series.
                Trend: The increasing or decreasing value in the series.
                Seasonality: The repeating short-term cycle in the series.
                Noise: The random variation in the series.
                '''

                # cryptoData[['Close']].plot(title='Close Price')
                decompose_result = seasonal_decompose(cryptoData['Close'], model ='multiplicative')
                # decompose_result.plot()

                # Set the value of Alpha and define m (Time Period)
                m = 365
                alpha = 1 / (2 * m)
                # Splitting the dataset to train and test
                train_df = cryptoData[:int(0.8 * cryptoData['Close'].shape[0])]
                test_df = cryptoData[int(0.8 * cryptoData['Close'].shape[0]):]

                # Plotting Train and Test Dataset
                train_df['Close'].plot(legend=True, label='Train')
                test_df['Close'].plot(legend=True, label='Test', figsize=(9, 4))
                plt.title('Holt Winter Double Exp Smoothing (Additive + Multiplicative)')

                # Additive Trend
                fitted_model = ExponentialSmoothing(train_df['Close'],trend='add').fit()
                # print(type(train_df.index))
                prediction_df=fitted_model.forecast(test_df.shape[0])
                # print(test_df)

                # Prediction using Additive
                prediction_df.plot(legend=True, label='Prediction Add')
                # cryptoData[['Close', 'HWS1']].plot(title='Holt Winters Single Exponential Smoothing')

                performance = PerformanceMetrics()
                mse = performance.mse(test_df['Close'], prediction_df)
                print("Add Double MSE")
                print(mse)


                # Multiplicative Trend
                # cryptoData['HWS2-Mult'] = ExponentialSmoothing(cryptoData['Close'], trend='mul').fit(smoothing_level=alpha, optimized=False, use_brute=True).fittedvalues
                fitted_model = ExponentialSmoothing(train_df['Close'], trend='mul').fit()
                # print(type(train_df.index))
                prediction_df = fitted_model.forecast(test_df.shape[0])
                print(test_df)

                # Predicting Using Multiplicative
                prediction_df.plot(legend=True, label='Prediction Mult')
                # cryptoData[['Close', 'HWS1']].plot(title='Holt Winters Single Exponential Smoothing')



                # cryptoData[['Close', 'HWS2-Add','HWS2-Mult']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend');
                plt.show()

                performance = PerformanceMetrics()
                mse = performance.mse(test_df['Close'], prediction_df)
                print("Mult Double MSE")
                print(mse)


    def holt_winters_triple_exp(self):

        # open datasets file
        for f in os.listdir('D:\\pycharm\\Goodman_Predict_Software/Dataset'):
            print(f)
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'D:\\pycharm\\Goodman_Predict_Software/Dataset/{f}', index_col=[0],
                                         parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.iloc[::-1]
                cryptoData = cryptoData.drop('Currency', axis=1)

                # print(cryptoData.shape)
                # print(cryptoData.head())

                # plotting the original data
                ''' 
                Here we Look for:
                Level: The average value in the series.
                Trend: The increasing or decreasing value in the series.
                Seasonality: The repeating short-term cycle in the series.
                Noise: The random variation in the series.
                '''

                # cryptoData[['Close']].plot(title='Close Price')
                decompose_result = seasonal_decompose(cryptoData['Close'], model='multiplicative')
                # decompose_result.plot()

                # Set the value of Alpha and define m (Time Period)
                m = 365
                alpha = 1 / (2 * m)
                # Splitting the dataset to train and test
                train_df = cryptoData[:int(0.1 * cryptoData['Close'].shape[0])]
                traindf_list=[]
                traindf_list.append(train_df)
                for p in range(2,8):
                    train_df = cryptoData[int((p-1)/10 * cryptoData['Close'].shape[0]):int(p/10 * cryptoData['Close'].shape[0])]
                    traindf_list.append(train_df)

                test_df = cryptoData[int(0.8 * cryptoData['Close'].shape[0]):]

                # Plotting Train and Test Dataset
                train_df['Close'].plot(legend=True, label='Train')
                test_df['Close'].plot(legend=True, label='Test', figsize=(9, 4))
                plt.title('Holt Winter Double Exp Smoothing (Additive + Multiplicative)')

                # Additive Trend
                fitted_model = ExponentialSmoothing(traindf_list[0]['Close'], trend='add', seasonal='add',seasonal_periods=365).fit()
                # print(type(train_df.index))
                prediction_df = fitted_model.forecast(test_df.shape[0])
                # print(test_df)

                # Prediction using Additive
                prediction_df.plot(legend=True, label='Prediction Add')
                # cryptoData[['Close', 'HWS1']].plot(title='Holt Winters Single Exponential Smoothing')
                performance = PerformanceMetrics()
                mse = performance.mse(test_df['Close'], prediction_df)
                print("Add Triple MSE")
                print(mse)

                # Multiplicative Trend
                # cryptoData['HWS2-Mult'] = ExponentialSmoothing(cryptoData['Close'], trend='mul').fit(smoothing_level=alpha, optimized=False, use_brute=True).fittedvalues
                fitted_model = ExponentialSmoothing(train_df['Close'], trend='mul', seasonal='mul',seasonal_periods=365).fit()
                # print(type(train_df.index))
                prediction_df = fitted_model.forecast(test_df.shape[0])
                print(test_df)

                # Predicting Using Multiplicative
                prediction_df.plot(legend=True, label='Prediction Mult')
                # cryptoData[['Close', 'HWS1']].plot(title='Holt Winters Single Exponential Smoothing')

                # cryptoData[['Close', 'HWS2-Add','HWS2-Mult']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend');
                plt.show()

                performance=PerformanceMetrics()
                mse=performance.mse(test_df['Close'],prediction_df)
                print("Mult Triple MSE")
                print(mse)


# model=PredictionModels()
# model.holt_winters_single_exp()
# model.holt_winters_double_exp()
# model.holt_winters_triple_exp()

print("hello")