from emb_shapley_azure import emb_shapley_azure
import prophet_predict_lib as ppl
import pandas as pd
import matplotlib.pyplot as plt
import os

fair_co2_path = os.environ.get('FAIR_CO2')
figures_dir = f'{fair_co2_path}/figures'

def forecast(inputs):
    t, eval_granularity, df_demand, resource, node, df_ci_5_min, df_ci_hour = inputs
    i = t // eval_granularity
    day = t / (60 * 60 * 24)
    history_length = (t - df_demand['timestamp'].min()) / (60 * 60 * 24)
    predict_length = (df_demand['timestamp'].max() - t) / (60 * 60 * 24)
    predict_start = (t - df_demand['timestamp'].min()) / (60 * 60 * 24)
    # Split df_demand into before and after t
    df_before = df_demand[df_demand['timestamp'] < t]
    df_after = df_demand[df_demand['timestamp'] >= t]
    df_before = df_before[['timestamp', 'cpu allocated']]
    df_after = df_after[['timestamp', 'cpu allocated']]
    df_before['timestamp'] = df_before['timestamp'] / (60 * 60 * 24)
    df_after['timestamp'] = df_after['timestamp'] / (60 * 60 * 24)
    # Split df_ci into before and after t
    df_ci_5_min_before = df_ci_5_min[df_ci_5_min['timestamp'] < t]
    df_ci_5_min_after = df_ci_5_min[df_ci_5_min['timestamp'] >= t]
    df_ci_hour_before = df_ci_hour[df_ci_hour['timestamp'] < t]
    df_ci_hour_after = df_ci_hour[df_ci_hour['timestamp'] >= t]
    df_ci_5_min['timestamp'] = df_ci_5_min['timestamp'] / (60 * 60 * 24)
    df_ci_hour['timestamp'] = df_ci_hour['timestamp'] / (60 * 60 * 24)
    df_ci_5_min_before['timestamp'] = df_ci_5_min_before['timestamp'] / (60 * 60 * 24)
    df_ci_5_min_after['timestamp'] = df_ci_5_min_after['timestamp'] / (60 * 60 * 24)
    df_ci_hour_before['timestamp'] = df_ci_hour_before['timestamp'] / (60 * 60 * 24)
    df_ci_hour_after['timestamp'] = df_ci_hour_after['timestamp'] / (60 * 60 * 24)

    # Save the actual ci data to CSV
    df_ci_hour.to_csv(f'{fair_co2_path}/forecast/actual_ci_hour_' + str(i) + '.csv', index=False)
    df_ci_5_min.to_csv(f'{fair_co2_path}/forecast/actual_ci_5_min_' + str(i) + '.csv', index=False)

    forecast_demand = ppl.predict(time_series_file, history_length, predict_length, predict_start, resource=resource, tuned=True)
    forecast_demand = forecast_demand[['timestamp', 'cpu allocated']]
    forecast_ci_5_min, forecast_ci_hour = emb_shapley_azure(forecast_demand, node, resource)
    # Save the forecasted demand data to CSV
    forecast_demand.to_csv(f'{fair_co2_path}/forecast/forecast_demand_' + str(i) + '.csv', index=False)
    forecast_demand['timestamp'] = forecast_demand['timestamp'] / (60 * 60 * 24)
    # Use forecast to predict carbon intensity

    forecast_ci_5_min['timestamp'] = forecast_ci_5_min['timestamp'] / (60 * 60 * 24)
    forecast_ci_hour['timestamp'] = forecast_ci_hour['timestamp'] / (60 * 60 * 24)

    # Save the forecasted carbon intensity data to CSV
    forecast_ci_hour.to_csv(f'{fair_co2_path}/forecast/forecast_ci_hour_' + str(i) + '.csv', index=False)
    forecast_ci_5_min.to_csv(f'{fair_co2_path}/forecast/forecast_ci_5_min_' + str(i) + '.csv', index=False)

    # Split forecast_demand into before and after t
    forecast_demand_after = forecast_demand[forecast_demand['timestamp'] >= day]

    plt.figure(figsize=(5, 2))
    plt.plot(df_before['timestamp'], df_before['cpu allocated'], linewidth=1)
    plt.plot(df_after['timestamp'], df_after['cpu allocated'], linewidth=1)
    # plt.plot(untuned_forecast_demand_after['timestamp'], untuned_forecast_demand_after['cpu allocated'], linestyle='dotted', linewidth=1)
    plt.plot(forecast_demand_after['timestamp'], forecast_demand_after['cpu allocated'], linewidth=1)
    # Vertical line at t
    # plt.axvline(x=t, color='r', linestyle='--')
    plt.yticks([])
    plt.xticks(fontsize=12)
    # x-axis break from 7 days to 14 days
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('CPUs Allocated', fontsize=12)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.25)
    plt.legend(['Actual (Past)', 'Actual (Future)', 'Forecast'], fontsize=12)
    plt.savefig(f'{fair_co2_path}/figures/5_forecast_demand.png', dpi=300)
    plt.close()

def forecast_plot():
    forecast_ci_5_min = pd.read_csv(f'{fair_co2_path}/forecast/forecast_ci_5_min_21.csv')
    df_ci_5_min = pd.read_csv(f'{fair_co2_path}/forecast/actual_ci_5_min_21.csv')
    average_ci_5_min = df_ci_5_min['embodied ci (gCO2eq/core-second)'].mean()
    day = 21

    baseline_ci = pd.DataFrame()
    baseline_ci['timestamp'] = df_ci_5_min['timestamp']
    baseline_ci['embodied ci (gCO2eq/core-second)'] = average_ci_5_min  

    # Calculate errors
    forecast_ci_error = pd.DataFrame()
    forecast_ci_error['timestamp'] = forecast_ci_5_min['timestamp']
    forecast_ci_error['embodied ci (gCO2eq/core-second)'] = forecast_ci_5_min['embodied ci (gCO2eq/core-second)'] - df_ci_5_min['embodied ci (gCO2eq/core-second)']

    baseline_ci_error = pd.DataFrame()
    baseline_ci_error['timestamp'] = df_ci_5_min['timestamp']
    baseline_ci_error['embodied ci (gCO2eq/core-second)'] = average_ci_5_min - df_ci_5_min['embodied ci (gCO2eq/core-second)']

    forecast_ci_error_relative = pd.DataFrame()
    forecast_ci_error_relative['timestamp'] = forecast_ci_5_min['timestamp']
    forecast_ci_error_relative['error (%)'] = forecast_ci_error['embodied ci (gCO2eq/core-second)'] / df_ci_5_min['embodied ci (gCO2eq/core-second)'] * 100

    baseline_ci_error_relative = pd.DataFrame()
    baseline_ci_error_relative['timestamp'] = df_ci_5_min['timestamp']
    baseline_ci_error_relative['error (%)'] = baseline_ci_error['embodied ci (gCO2eq/core-second)'] / df_ci_5_min['embodied ci (gCO2eq/core-second)'] * 100

    # Plot two subplots stacked
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    axs[0].plot(df_ci_5_min['timestamp'], df_ci_5_min['embodied ci (gCO2eq/core-second)'], linewidth=1.5)
    axs[0].plot(forecast_ci_5_min['timestamp'], forecast_ci_5_min['embodied ci (gCO2eq/core-second)'], linewidth=1.5)
    # axs[0].plot(baseline_ci['timestamp'], baseline_ci['embodied ci (gCO2eq/core-second)'], linewidth=1.5)
    # Vertical line at t
    axs[0].axvline(x=day, color='r', linestyle='--')
    # y axis labels
    axs[0].set_ylabel('Embodied CI\n(gCO2eq/core-s)', fontsize=14)
    # set axis tick fontsize
    axs[0].yaxis.set_tick_params(labelsize=14)

    # Plot relative error
    axs[1].plot(forecast_ci_error_relative['timestamp'], forecast_ci_error_relative['error (%)'], linewidth=1.5)
    # axs[1].plot(baseline_ci_error_relative['timestamp'], baseline_ci_error_relative['error (%)'], linewidth=1.5)
    # Vertical line at t
    axs[1].axvline(x=day, color='r', linestyle='--')
    # set axis tick fontsize
    axs[1].yaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    # axis labels
    axs[1].set_ylabel('Error (%)', fontsize=14)
    axs[1].set_xlabel('Time (Days)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fair_co2_path}/figures/11_forecast_ci_error_5_min.png')
    plt.close()

    # Find average error
    forecast_ci_error_relative_abs = forecast_ci_error_relative['error (%)'].abs()
    average_error_abs = forecast_ci_error_relative_abs.mean()
    print('Mean Absolute Percentage Error = ', average_error_abs)
    print('Max Absolute Percentage Error = ', forecast_ci_error_relative_abs.max())
    # Find average error after day
    forecast_ci_error_relative_abs_after_day = forecast_ci_error_relative_abs[forecast_ci_error_relative['timestamp'] > day]
    average_error_abs_after_day = forecast_ci_error_relative_abs_after_day.mean()
    print('Mean Absolute Percentage Error after day = ', average_error_abs_after_day)

    # Find RMSE
    rmse = (forecast_ci_error['embodied ci (gCO2eq/core-second)']**2).mean()**0.5
    print('RMSE = ', rmse)
    # Find RMSE after day
    rmse_after_day = (forecast_ci_error['embodied ci (gCO2eq/core-second)']**2).mean()**0.5
    print('RMSE after day = ', rmse_after_day)

if __name__ == "__main__":
    time_series_file = f'{fair_co2_path}/forecast/azure_time_series.csv'

    eval_granularity = 60 * 60 * 24 # 1 day

    resource = 'cpu'
    node = 'clr'

    # Load the time series data
    df_demand = pd.read_csv(time_series_file)

    # Generate the carbon intensity data from the actual demand trace
    df_ci_5_min, df_ci_hour = emb_shapley_azure(df_demand, node, resource)

    # Create dataframes for error metrics
    df_demand_error = pd.DataFrame(columns=['timestamp', 'mape (hour)', 'mae (hour)', 'rmse (hour)', 'mape (day)', 'mae (day)', 'rmse (day)', 'mape (week)', 'mae (week)', 'rmse (week)'])
    df_ci_error = pd.DataFrame(columns=['timestamp', 'mape (hour)', 'mae (hour)', 'rmse (hour)', 'mape (day)', 'mae (day)', 'rmse (day)', 'mape (week)', 'mae (week)', 'rmse (week)'])
    
    # Forecast the demand and carbon intensity
    inputs = [df_demand['timestamp'].min() + eval_granularity * 21, eval_granularity, df_demand, resource, node, df_ci_5_min, df_ci_hour]
    forecast(inputs)
    forecast_plot()

  

