import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import datetime

fair_co2_path = os.environ.get('FAIR_CO2')

def get_wl_configs(wl_df, grid_ci, cpu_ci, mem_ci, config_param_cols, energy_col, runtime_col, cpu_col, mem_col, throughput=False):
    # Find the configuration with the lowest carbon intensity
    df = wl_df.copy()
    df['Operational CF (gCO2eq)'] = df[energy_col] * grid_ci
    df['Embodied CPU CF (gCO2eq)'] = df[cpu_col] * cpu_ci 
    df['Embodied Memory CF (gCO2eq)'] = df[mem_col] * mem_ci
    df['Embodied CF (gCO2eq)'] = df['Embodied CPU CF (gCO2eq)'] + df['Embodied Memory CF (gCO2eq)']
    df['Total CF (gCO2eq)'] = df['Operational CF (gCO2eq)'] + df['Embodied CPU CF (gCO2eq)'] + df['Embodied Memory CF (gCO2eq)']
    # Throw error if are NaN values in the dataframe
    if df.isnull().values.any():
        print(f'Grid CI: {grid_ci}, CPU CI: {cpu_ci}, Memory CI: {mem_ci}')
        print(df['Operational CF (gCO2eq)'])
        print(df['Embodied CPU CF (gCO2eq)'])
        print(df['Embodied Memory CF (gCO2eq)'])
        print(df['Embodied CF (gCO2eq)'])
        print(df['Total CF (gCO2eq)'])
        raise ValueError('There are NaN values in the dataframe')
    # Minimum CF configuration
    min_cf_idx = df['Total CF (gCO2eq)'].idxmin()
    # Minimum runtime configuration
    if throughput == False:
        min_runtime_idx = df[runtime_col].idxmin()
    else:
        min_runtime_idx = df[runtime_col].idxmax()
    # Minimum energy configuration
    min_energy_idx = df[energy_col].idxmin()
    # Minimum Embodied CF configuration
    min_embodied_cf_idx = df['Embodied CF (gCO2eq)'].idxmin()

    min_cf_config = df.loc[min_cf_idx, config_param_cols].to_list()
    min_runtime_config = df.loc[min_runtime_idx, config_param_cols].to_list()
    min_energy_config = df.loc[min_energy_idx, config_param_cols].to_list()
    min_embodied_cf_config = df.loc[min_embodied_cf_idx, config_param_cols].to_list()

    min_cf_cf = df.at[min_cf_idx, 'Total CF (gCO2eq)']
    min_runtime_cf = df.at[min_runtime_idx, 'Total CF (gCO2eq)']
    min_energy_cf = df.at[min_energy_idx, 'Total CF (gCO2eq)']
    min_embodied_cf = df.at[min_embodied_cf_idx, 'Total CF (gCO2eq)']

    min_cf_emb_cf = df.at[min_cf_idx, 'Embodied CF (gCO2eq)']
    min_runtime_emb_cf = df.at[min_runtime_idx, 'Embodied CF (gCO2eq)']
    min_energy_emb_cf = df.at[min_energy_idx, 'Embodied CF (gCO2eq)']
    min_embodied_emb_cf = df.at[min_embodied_cf_idx, 'Embodied CF (gCO2eq)']

    min_cf_operational_cf = df.at[min_cf_idx, 'Operational CF (gCO2eq)']
    min_runtime_operational_cf = df.at[min_runtime_idx, 'Operational CF (gCO2eq)']
    min_energy_operational_cf = df.at[min_energy_idx, 'Operational CF (gCO2eq)']
    min_embodied_operational_cf = df.at[min_embodied_cf_idx, 'Operational CF (gCO2eq)']

    return [min_cf_cf, min_cf_config, min_cf_emb_cf, min_cf_operational_cf], [min_runtime_cf, min_runtime_config, min_runtime_emb_cf, min_runtime_operational_cf], [min_energy_cf, min_energy_config, min_energy_emb_cf, min_energy_operational_cf], [min_embodied_cf, min_embodied_cf_config, min_embodied_emb_cf, min_embodied_operational_cf]

def truncate_grid_ci(grid_ci_df, start_time='2023-10-01 00:00:00', end_time='2023-10-31 23:59:59'):
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    grid_ci_df['Datetime (UTC)'] = pd.to_datetime(grid_ci_df['Datetime (UTC)'])
    grid_ci_df = grid_ci_df[(grid_ci_df['Datetime (UTC)'] >= start_time) & (grid_ci_df['Datetime (UTC)'] < end_time)]
    return grid_ci_df['Carbon Intensity gCO₂eq/kWh (LCA)'].to_list()

def dynamic_wl_config(wl_df, grid_ci, cpu_ci, mem_ci, config_param_cols, energy_col, runtime_col, cpu_col, mem_col, start_time, end_time, time_interval, file_to_save):
    df = pd.DataFrame()
    J_to_kWh = 1 / 3600000
    time = np.arange(start_time, end_time, time_interval)
    # Set time as index
    df['Time (s)'] = time
    df['grid CI (gCO2eq/kWh)'] = np.zeros(len(time))
    df['grid CI (gCO2eq/J)'] = np.zeros(len(time))
    df['cpu CI (gCO2eq/core-second)'] = np.zeros(len(time))
    df['mem CI (gCO2eq/GB-second)'] = np.zeros(len(time))
    df['Min CF Config'] = ''
    df['Min Runtime Config'] = ''
    df['Min Energy Config'] = ''
    df['Min Embodied CF Config'] = ''
    # Check lengths of grid_ci, cpu_ci, and mem_ci
    if len(grid_ci) != len(cpu_ci) or len(grid_ci) != len(mem_ci) or len(time) != len(grid_ci):
        print(len(grid_ci), len(cpu_ci), len(mem_ci), len(time))
        raise ValueError('Lengths of grid_ci, cpu_ci, and mem_ci must be the same')

    for i, t in enumerate(time):
        # Get the workload configuration at time t
        wl_config = get_wl_configs(wl_df.copy(), grid_ci[i] * J_to_kWh, cpu_ci[i], mem_ci[i], config_param_cols, energy_col, runtime_col, cpu_col, mem_col)
        # Add the CI values to the dataframe
        df.at[i, 'grid CI (gCO2eq/kWh)'] = grid_ci[i] 
        df.at[i, 'grid CI (gCO2eq/J)'] = grid_ci[i] * J_to_kWh
        df.at[i, 'cpu CI (gCO2eq/core-second)'] = cpu_ci[i]
        df.at[i, 'mem CI (gCO2eq/GB-second)'] = mem_ci[i]
        # Add the workload configuration to the dataframe
        df.at[i, 'Min CF Config'] = wl_config[0][1]
        df.at[i, 'Min Runtime Config'] = wl_config[1][1]
        df.at[i, 'Min Energy Config'] = wl_config[2][1]
        df.at[i, 'Min Embodied CF Config'] = wl_config[3][1]
        # Add the CF values to the dataframe
        df.at[i, 'Min CF CF (gCO2eq)'] = wl_config[0][0]
        df.at[i, 'Min Runtime CF (gCO2eq)'] = wl_config[1][0]
        df.at[i, 'Min Energy CF (gCO2eq)'] = wl_config[2][0]
        df.at[i, 'Min Embodied CF (gCO2eq)'] = wl_config[3][0]
        df.at[i, 'Min CF Embodied CF (gCO2eq)'] = wl_config[0][2]
        df.at[i, 'Min Runtime Embodied CF (gCO2eq)'] = wl_config[1][2]
        df.at[i, 'Min Energy Embodied CF (gCO2eq)'] = wl_config[2][2]
        df.at[i, 'Min Embodied Embodied CF (gCO2eq)'] = wl_config[3][2]
        df.at[i, 'Min CF Operational CF (gCO2eq)'] = wl_config[0][3]
        df.at[i, 'Min Runtime Operational CF (gCO2eq)'] = wl_config[1][3]
        df.at[i, 'Min Energy Operational CF (gCO2eq)'] = wl_config[2][3]
        df.at[i, 'Min Embodied Operational CF (gCO2eq)'] = wl_config[3][3]

    df['Min CF % Improvement'] = (df['Min Runtime CF (gCO2eq)'] - df['Min CF CF (gCO2eq)']) / df['Min Runtime CF (gCO2eq)'] * 100
    df['Min Energy % Improvement'] = (df['Min Runtime CF (gCO2eq)'] - df['Min Energy CF (gCO2eq)']) / df['Min Runtime CF (gCO2eq)'] * 100
    df['Min Embodied % Improvement'] = (df['Min Runtime CF (gCO2eq)'] - df['Min Embodied CF (gCO2eq)']) / df['Min Runtime CF (gCO2eq)'] * 100

    # Save the dataframe to a csv file
    df.to_csv(file_to_save)

    return df

if __name__ == '__main__':
    line_width = 2
    label_font_size = 13
    title_font_size = 14
    legend_font_size = 11
    tick_font_size = 14
    # Time interval in seconds
    time_interval = 10
    # Workload dataframe
    faiss_df = pd.read_csv(f'{fair_co2_path}/workload-optimization/results/faiss/faiss_processed.csv')
    faiss_df['Average Latency (s)'] = faiss_df['Average Latency (ms)'] / 1000
    faiss_df['P99 Latency (s)'] = faiss_df['99th Percentile (ms)'] / 1000
    # Grid CI sources
    grid_ci_sources = ['GB', 'US_VA', 'US_CA', 'Sweden_SC', 'Canada_ON', 'India_S']
    # Carbon intensity values
    azure_cpu_ci = pd.read_csv(f'{fair_co2_path}/carbon-intensity/azure_cpu_ci_hourly.csv')
    azure_mem_ci = pd.read_csv(f'{fair_co2_path}/carbon-intensity/azure_mem_ci_hourly.csv')
    azure_cpu_ci = azure_cpu_ci['0'].to_list()
    azure_mem_ci = azure_mem_ci['0'].to_list()
    azure_cpu_ci = azure_cpu_ci[:30*24]
    azure_mem_ci = azure_mem_ci[:30*24]
    # Configuration parameter columns
    spark_config_param_cols = ['threads', 'memory']
    faiss_config_param_cols = ['Index', 'CPU Cores', 'Batch Size']
    # Energy column
    spark_energy_col = 'Energy per Round (J)'
    faiss_energy_col = 'Total Energy per Query (Joules)'
    # Runtime column
    spark_runtime_col = 'total_runtime'
    # faiss_runtime_col = 'Average Latency (ms)'
    faiss_runtime_col = '99th Percentile (ms)'
    # CPU core-seconds column
    spark_cpu_col = 'CPU-seconds'
    faiss_cpu_col = 'CPU-Seconds Per Query'
    # Memory column
    spark_mem_col = 'Memory-seconds'
    faiss_mem_col = 'Memory Usage per Query (GB-second)'

    # Time interval
    start_time = '2023-05-01 00:00:00'
    end_time = '2023-05-30 23:59:59'
    start_time_seconds = 0
    end_time_seconds = 30 * 24 * 60 * 60
    time_interval = 3600
    # File to save the results
    wl_df = faiss_df
    # Filter out the rows with runtime greater than x seconds
    wl_df = wl_df[wl_df['P99 Latency (s)'] <= 1.9]
    config_param_cols = faiss_config_param_cols
    energy_col = faiss_energy_col
    runtime_col = faiss_runtime_col
    cpu_col = faiss_cpu_col
    mem_col = faiss_mem_col
    node = 'clr'
    grid_ci = pd.read_csv(f'{fair_co2_path}/carbon-intensity/US-CAL-CISO_2023_hourly.csv')
    grid_ci = truncate_grid_ci(grid_ci, start_time, end_time)
    cpu_ci = pd.read_csv(f'{fair_co2_path}/carbon-intensity/azure_cpu_ci_hourly.csv')
    mem_ci = pd.read_csv(f'{fair_co2_path}/carbon-intensity/azure_mem_ci_hourly.csv')
    cpu_ci = cpu_ci['0'].to_list()
    mem_ci = mem_ci['0'].to_list()
    cpu_ci = cpu_ci[:30*24]
    mem_ci = mem_ci[:30*24]
    print("Processing workload: FAISS, Grid CI source: US_CA, Embodied CI source: Azure 2017")
    file_to_save = f'{fair_co2_path}/workload-optimization/results/faiss/faiss_US_CA_Azure2017_dynamic_wl_config.csv'
    df = dynamic_wl_config(wl_df, grid_ci, cpu_ci, mem_ci, config_param_cols, energy_col, runtime_col, cpu_col, mem_col, start_time_seconds, end_time_seconds, time_interval, file_to_save)

    # Plot the results
    plot_start_time = 7*24*60*60
    plot_end_time = 14*24*60*60
    plot_start_date = 0
    plot_end_date = (plot_end_time - plot_start_time) /(24*60*60)
    df = df[(df['Time (s)'] >= plot_start_time) & (df['Time (s)'] < plot_end_time)]
    df['Time (s)'] = df['Time (s)'] - plot_start_time
    df['Time (days)'] = df['Time (s)'] / (24*60*60)
    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    ax[0].set_xlim(plot_start_date, plot_end_date)
    ax[1].set_xlim(plot_start_date, plot_end_date)
    ax[2].set_xlim(plot_start_date, plot_end_date)
    ax[3].set_xlim(plot_start_date, plot_end_date)
    ax_conf_2 = ax[0].twinx()

    # Plot dotted line if index is hnsw and plot solid line if index is ivf
    ax[0].plot(df['Time (days)'], df['Min CF Config'].apply(lambda x: x[1]), label='CPU Cores', linestyle='solid', color='green')
    # ax[0].plot(df['Time (days)'], df['Min Runtime Config'].apply(lambda x: x[1]), label='Min Runtime Config CPU', linestyle='dotted', color='green')
    # ax[0].plot(df['Time (days)'], df['Min Energy Config'].apply(lambda x: x[1]), label='Min Energy Config CPU', linestyle='dotted', color='blue')
    # ax[0].plot(df['Time (days)'], df['Min Embodied CF Config'].apply(lambda x: x[1]), label='Min Embodied CF Config CPU', linestyle='dotted', color='purple')
    ax[0].set_ylabel('CPU Cores', fontsize = label_font_size)
    ax[0].set_title('Carbon-Optimal Workload Configuration', fontsize = title_font_size)
    ax_conf_2.plot(df['Time (days)'], df['Min CF Config'].apply(lambda x: x[2]), label='Batch Size', linestyle='solid', color='orange')
    # ax_conf_2.plot(df['Time (days)'], df['Min Runtime Config'].apply(lambda x: x[2]), label='Min Runtime Config Batch Size', linestyle='dotted', color='orange')
    # ax_conf_2.plot(df['Time (days)'], df['Min Energy Config'].apply(lambda x: x[2]), label='Min Energy Config Batch Size', linestyle='dotted', color='blue')
    # ax_conf_2.plot(df['Time (days)'], df['Min Embodied CF Config'].apply(lambda x: x[2]), label='Min Embodied CF Config Batch Size', linestyle='dotted', color='purple')
    ax_conf_2.set_ylabel('Batch Size', fontsize = label_font_size)
    # Shade in the area where the index is hnsw
    ax[0].fill_between(df['Time (days)'], 0, 100, where=df['Min CF Config'].apply(lambda x: x[0]) == 'hnsw', color='blue', alpha=0.3)
    ax[0].fill_between(df['Time (days)'], 0, 100, where=df['Min CF Config'].apply(lambda x: x[0]) == 'ivf', color='fuchsia', alpha=0.3)
    handles_0, labels_0 = ax[0].get_legend_handles_labels()
    handles_2, labels_2 = ax_conf_2.get_legend_handles_labels()
    ax[0].legend(handles_0 + handles_2, labels_0 + labels_2, fontsize = legend_font_size, ncol=2, loc='lower left')     
    # Add extra legend for index type   
    ax_dummy = ax[0].twinx()
    ax_dummy.axis('off')
    ax_dummy.legend(handles=[Patch(color='blue', alpha=0.3), Patch(color='fuchsia', alpha=0.3)], labels=['HNSW', 'IVF'], loc='lower right', fontsize = legend_font_size, ncol=2)

    # 2nd plot CF
    ax[1].plot(df['Time (days)'], df['Min CF % Improvement'], label='Min CF CF', linestyle='solid', color='green')
    # ax[1].plot(df['Time (days)'], df['Min Energy % Improvement'], label='Min Energy CF', linestyle='dotted', color='blue')
    # ax[2].plot(df['Time (days)'], df['Min Embodied CF (gCO2eq)'], label='Min Embodied CF', linestyle='dotted', color='purple')
    ax[1].set_ylabel('CF Reduction (%)', fontsize = label_font_size)
    ax[1].set_title('Carbon Savings vs. Performance-Optimized Configuration', fontsize = title_font_size)

    # 3rd plot grid CI
    ax[2].plot(df['Time (days)'], df['grid CI (gCO2eq/kWh)'], label='Grid CI (gCO2eq/kWh)')
    ax[2].set_ylabel('gCO$_{2}$e/kWh', fontsize = label_font_size)
    ax[2].set_title('Grid Carbon Intensity', fontsize = title_font_size)

    # 4th plot embodied CI (CPU on left y-axis, memory on right y-axis)
    ax[3].plot(df['Time (days)'], df['cpu CI (gCO2eq/core-second)'], label='CPU')
    ax_mem_ci = ax[3].twinx()
    ax_mem_ci.plot(df['Time (days)'], df['mem CI (gCO2eq/GB-second)'], label='Memory', color='red')
    ax_mem_ci.set_ylabel('Mem. Emb. CI\n(gCO$_{2}$e/GB-s)', fontsize = label_font_size)
    ax[3].set_ylabel('CPU Emb. CI\n(gCO$_{2}$eq/core-s)', fontsize = label_font_size)
    ax[3].set_title('Embodied Carbon Intensity', fontsize = title_font_size)
    ax[3].set_xlabel('Time (days)', fontsize=14)
    handles_cpu, labels_cpu = ax[3].get_legend_handles_labels()
    handles_mem, labels_mem = ax_mem_ci.get_legend_handles_labels()
    ax_mem_ci.legend(handles_cpu + handles_mem, labels_cpu + labels_mem, fontsize = legend_font_size)

    # Change the margins
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.09)
    plt.savefig(f'{fair_co2_path}/figures/13_faiss_US_CA_Azure2017_dynamic_wl_config.pdf', dpi=300)
    plt.close()