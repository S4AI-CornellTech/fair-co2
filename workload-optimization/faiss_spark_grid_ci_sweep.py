import pandas as pd
import numpy as np
import os
from gen_dyn_wl_figure import get_wl_configs

fair_co2_path = os.environ.get('FAIR_CO2')
figures_dir = f'{fair_co2_path}/figures'
data_dir = f'{fair_co2_path}/workload-optimization/results'

grid_ci_list = range(0, 1010, 10)
lifetime = 4 * 365 * 24 * 60 * 60# seconds

spark_df = pd.read_csv(f'{data_dir}/spark/spark.csv')
faiss_df = pd.read_csv(f'{data_dir}/faiss/faiss_processed.csv')
spark_df['Energy per Round (J)'] = spark_df['Average CPU Energy per Round'] + spark_df['Average DRAM Energy per Round']
faiss_df['Average Latency (s)'] = faiss_df['Average Latency (ms)'] / 1000

# Configuration parameter columns
spark_config_param_cols = ['memory', 'threads']
faiss_config_param_cols = ['Index', 'CPU Cores', 'Batch Size']

# Energy column
spark_energy_col = 'Energy per Round (J)'
faiss_energy_col = 'Total Energy per Query (Joules)'

# Runtime column
spark_runtime_col = 'total_runtime'
faiss_runtime_col = 'Average Latency (ms)'

# CPU core-seconds column
spark_cpu_col = 'CPU-seconds'
faiss_cpu_col = 'CPU-Seconds Per Query'

# Memory column
spark_mem_col = 'Memory-seconds'
faiss_mem_col = 'Memory Usage per Query (GB-second)'

df = pd.DataFrame()
df['Grid CI (gCO2eq/kWh)'] = grid_ci_list
df['Grid CI (gCO2eq/J)'] = np.zeros(len(grid_ci_list))
df['cpu CI (gCO2eq/core-second)'] = np.zeros(len(grid_ci_list))
df['mem CI (gCO2eq/GB-second)'] = np.zeros(len(grid_ci_list))
df['Min CF Config'] = ''
df['Min Runtime Config'] = ''
df['Min Energy Config'] = ''
df['Min Embodied CF Config'] = ''
throughput = False

for workload in ['faiss', 'spark']:
    if workload == 'faiss':
        wl_df = faiss_df.copy()
        config_param_cols = faiss_config_param_cols
        energy_col = faiss_energy_col
        runtime_col = faiss_runtime_col
        cpu_col = faiss_cpu_col
        mem_col = faiss_mem_col
        file_to_save = f'{data_dir}/faiss/faiss_grid_ci_sweep.csv'
        pickle_to_save = f'{data_dir}/faiss/faiss_grid_ci_sweep.pkl'
    elif workload == 'spark':
        wl_df = spark_df.copy()
        config_param_cols = spark_config_param_cols
        energy_col = spark_energy_col
        runtime_col = spark_runtime_col
        cpu_col = spark_cpu_col
        mem_col = spark_mem_col
        file_to_save = f'{data_dir}/spark/spark_grid_ci_sweep.csv'
        pickle_to_save = f'{data_dir}/spark/spark_grid_ci_sweep.pkl'

    imec_cpu_chip_cf = 20540 # gCO2eq
    ACT_cpu_chip_cf = 18530 # gCO2eq
    num_cores_per_cpu = 48
    num_cpus_per_node = 2
    cooling_cf = 24210 # gCO2eq
    gb_per_node = 192
    dram_cf = 146875 # gCO2eq 
    ssd_cf_per_gb =  160 # gCO2eq, from Dirty Secrets of SSDs paper
    ssd_cap_per_node = 480 # GB
    ssd_cf = ssd_cf_per_gb * ssd_cap_per_node # gCO2eq
    mb_cf = 109670 # gCO2eq
    chassis_cf = 34300 # gCO2eq
    peripheral_cf = 59170 # gCO2eq
    psu_cf = 30016 # gCO2eq
    idle_cpu_power = 35.49978333333334
    idle_dram_power = 2.8108166666666663

    cpu_chip_cf_per_cpu = imec_cpu_chip_cf
    cpu_cf = cpu_chip_cf_per_cpu * num_cpus_per_node # gCO2eq
    cpu_cf_per_core = cpu_cf / num_cores_per_cpu # gCO2eq

    cpu_and_cooling_cf = cpu_cf + cooling_cf # gCO2eq
    cpu_and_cooling_cf_per_core = cpu_and_cooling_cf / num_cores_per_cpu # gCO2eq

    dram_cf_per_gb =  dram_cf / gb_per_node# gCO2eq

    other_cf = mb_cf + chassis_cf + peripheral_cf + psu_cf + ssd_cf # gCO2eq
    node_cf = (cpu_and_cooling_cf + dram_cf + ssd_cf + mb_cf + chassis_cf + peripheral_cf + psu_cf)

    kWh_to_J = 3.6e6 # J/kWh
    mem_ci = (dram_cf + 0.5 * other_cf) / gb_per_node / lifetime # gCO2eq/(GB-hour)
    cpu_ci = (cpu_and_cooling_cf + 0.5 * other_cf) / (num_cores_per_cpu * num_cpus_per_node) / lifetime # gCO2eq/(core-hour)
    node_ci = node_cf / lifetime # gCO2eq/node-hour

    for i, grid_ci in enumerate(grid_ci_list):
        # Get the workload configuration at time t
        wl_config = get_wl_configs(wl_df.copy(), grid_ci / kWh_to_J, cpu_ci, mem_ci, config_param_cols, energy_col, runtime_col, cpu_col, mem_col, throughput=throughput)
        # Add the CI values to the dataframe
        df.at[i, 'Grid CI (gCO2eq/kWh)'] = grid_ci
        df.at[i, 'Grid CI (gCO2eq/J)'] = grid_ci / kWh_to_J
        df.at[i, 'cpu CI (gCO2eq/core-second)'] = cpu_ci
        df.at[i, 'mem CI (gCO2eq/GB-second)'] = mem_ci
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
        
    # Save the dataframe to a csv file
    df.to_csv(file_to_save)
    # Save the dataframe to a pickle file
    df.to_pickle(pickle_to_save)
