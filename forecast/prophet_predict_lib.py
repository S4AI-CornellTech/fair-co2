from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Predicts the future values of the time series data in the given file
# filename: the name of the file containing the time series data
# history_length: the length of the history to use for prediction in days
# predict_length: the length of the future to predict in days
# predict_start: the day to start predicting from

def predict(filename, history_length, predict_length, predict_start, resource='cpu', convert_from_timestamp=False, data_granularity=300, tuned = False): 
    df = pd.read_csv(filename)

    history_in_seconds = 60*60*24*history_length
    predict_in_seconds = 60*60*24*predict_length
    predict_start_seconds = 60*60*24*predict_start

    df_before = df[df['timestamp'] < predict_start_seconds]
    df_after = df[df['timestamp'] >= predict_start_seconds]

    df_condition = df_before[df_before['timestamp'] >= (predict_start_seconds - history_in_seconds)]
    df_before = df_before[df_before['timestamp'] < predict_start_seconds - history_in_seconds]
    df_predict_actual = df_after[df_after['timestamp'] < predict_start_seconds + predict_in_seconds]
    df_after = df_after[df_after['timestamp'] >= predict_start_seconds + predict_in_seconds]

    # Set timestamp 0 to be 2017-01-01
    timestamp_offset = 1483246800
    datetime_offset = pd.to_datetime(timestamp_offset, unit='s')
    df['timestamp'] = df['timestamp'] + timestamp_offset
    df_before['timestamp'] = df_before['timestamp'] + timestamp_offset
    df_after['timestamp'] = df_after['timestamp'] + timestamp_offset
    df_condition['timestamp'] = df_condition['timestamp'] + timestamp_offset
    df_predict_actual['timestamp'] = df_predict_actual['timestamp'] + timestamp_offset

    # Convert timestamp column from seconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df_before['timestamp'] = pd.to_datetime(df_before['timestamp'], unit='s')
    df_after['timestamp'] = pd.to_datetime(df_after['timestamp'], unit='s')
    df_condition['timestamp'] = pd.to_datetime(df_condition['timestamp'], unit='s')
    df_predict_actual['timestamp'] = pd.to_datetime(df_predict_actual['timestamp'], unit='s')

    # Rename columns to ds and y
    if (resource == 'cpu'):
        df = df.rename(columns={'timestamp': 'ds', 'cpu allocated': 'y'})
        df_before = df_before.rename(columns={'timestamp': 'ds', 'cpu allocated': 'y'})
        df_after = df_after.rename(columns={'timestamp': 'ds', 'cpu allocated': 'y'})
        df_condition = df_condition.rename(columns={'timestamp': 'ds', 'cpu allocated': 'y'})
        df_predict_actual = df_predict_actual.rename(columns={'timestamp': 'ds', 'cpu allocated': 'y'})
    elif (resource == 'mem'):
        df = df.rename(columns={'timestamp': 'ds', 'mem allocated (gb)': 'y'})
        df_before = df_before.rename(columns={'timestamp': 'ds', 'mem allocated (gb)': 'y'})
        df_after = df_after.rename(columns={'timestamp': 'ds', 'mem allocated (gb)': 'y'})
        df_condition = df_condition.rename(columns={'timestamp': 'ds', 'mem allocated (gb)': 'y'})
        df_predict_actual = df_predict_actual.rename(columns={'timestamp': 'ds', 'mem allocated (gb)': 'y'})

    m = Prophet()
    if tuned:
        m.add_seasonality(name='weekly', period=7, fourier_order=20, prior_scale=50)
        m.add_seasonality(name='daily', period=1, fourier_order=50, prior_scale=100)
    # m.add_seasonality(name='four-hourly', period=1/6, fourier_order=50, prior_scale=100)
    # m.add_seasonality(name='hourly', period=1/24, fourier_order=50, prior_scale=100)
    m.fit(df_condition)

    predict_df = m.make_future_dataframe(periods=int(predict_in_seconds/60), freq='min', include_history=False)

    forecast = m.predict(predict_df)

    # Convert timestamp column from seconds to days
    df['ds'] = (df['ds'] - datetime_offset).dt.total_seconds()
    df_before['ds'] = (df_before['ds'] - datetime_offset).dt.total_seconds()
    df_after['ds'] = (df_after['ds'] - datetime_offset).dt.total_seconds()
    df_condition['ds'] = (df_condition['ds'] - datetime_offset).dt.total_seconds()
    df_predict_actual['ds'] = (df_predict_actual['ds'] - datetime_offset).dt.total_seconds()
    forecast['ds'] = (forecast['ds'] - datetime_offset).dt.total_seconds()

    # Subsample the forecast to the granularity of the actual data
    forecast = forecast[forecast['ds'] % data_granularity == 0]

    if resource=='cpu':
        df = df.rename(columns={'ds': 'timestamp', 'y': 'cpu allocated'})
        df_before = df_before.rename(columns={'ds': 'timestamp', 'y': 'cpu allocated'})
        df_after = df_after.rename(columns={'ds': 'timestamp', 'y': 'cpu allocated'})
        df_condition = df_condition.rename(columns={'ds': 'timestamp', 'y': 'cpu allocated'})
        df_predict_actual = df_predict_actual.rename(columns={'ds': 'timestamp', 'y': 'cpu allocated'})
        forecast = forecast.rename(columns={'ds': 'timestamp', 'yhat': 'cpu allocated'})
    elif resource=='mem':
        df = df.rename(columns={'ds': 'timestamp', 'y': 'mem allocated (gb)'})
        df_before = df_before.rename(columns={'ds': 'timestamp', 'y': 'mem allocated (gb)'})
        df_after = df_after.rename(columns={'ds': 'timestamp', 'y': 'mem allocated (gb)'})
        df_condition = df_condition.rename(columns={'ds': 'timestamp', 'y': 'mem allocated (gb)'})
        df_predict_actual = df_predict_actual.rename(columns={'ds': 'timestamp', 'y': 'mem allocated (gb)'})
        forecast = forecast.rename(columns={'ds': 'timestamp', 'yhat': 'mem allocated (gb)'})

    # Concatenate the actual data with the forecast
    forecast = pd.concat([df_before, df_condition, forecast, df_after], ignore_index=True)

    return forecast

# Calculate the RMSE of the forecast
def rmse(forecast, actual):
    return np.sqrt(np.mean((forecast - actual)**2))

# Calculate the MAE of the forecast
def mae(forecast, actual):
    return np.mean(np.abs(forecast - actual))

# Calculate the MAPE of the forecast
def mape(forecast, actual):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# Return next hour, next day, and next week error metrics
def error_metrics(forecast_df, actual_df, timestamp, timestamp_col, value_col):
    forecast_hour = forecast_df[forecast_df[timestamp_col] == timestamp + 60*60]
    forecast_day = forecast_df[forecast_df[timestamp_col] == timestamp + 60*60*24]
    forecast_week = forecast_df[forecast_df[timestamp_col] == timestamp + 60*60*24*7]
    actual_hour = actual_df[actual_df[timestamp_col] == timestamp + 60*60]
    actual_day = actual_df[actual_df[timestamp_col] == timestamp + 60*60*24]
    actual_week = actual_df[actual_df[timestamp_col] == timestamp + 60*60*24*7]

    # Calculate error metrics
    mape_hour = mape(forecast_hour[value_col].values, actual_hour[value_col].values)
    mae_hour = mae(forecast_hour[value_col].values, actual_hour[value_col].values)
    rmse_hour = rmse(forecast_hour[value_col].values, actual_hour[value_col].values)

    mape_day = mape(forecast_day[value_col].values, actual_day[value_col].values)
    mae_day = mae(forecast_day[value_col].values, actual_day[value_col].values)
    rmse_day = rmse(forecast_day[value_col].values, actual_day[value_col].values)

    mape_week = mape(forecast_week[value_col].values, actual_week[value_col].values)
    mae_week = mae(forecast_week[value_col].values, actual_week[value_col].values)
    rmse_week = rmse(forecast_week[value_col].values, actual_week[value_col].values)

    return {'timestamp': timestamp, 'mape (hour)': mape_hour, 'mae (hour)': mae_hour, 'rmse (hour)': rmse_hour, 'mape (day)': mape_day, 'mae (day)': mae_day, 'rmse (day)': rmse_day, 'mape (week)': mape_week, 'mae (week)': mae_week, 'rmse (week)': rmse_week}