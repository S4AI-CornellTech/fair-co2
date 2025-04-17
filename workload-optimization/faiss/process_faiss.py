# Main script that runs all the processing scripts
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re

fair_co2_path = os.environ.get('FAIR_CO2')
faiss_results_dir = f'{fair_co2_path}/workload-optimization/results/faiss'
data_dir = faiss_results_dir

def create_folders(data_dir):
    # Make data folder
    os.makedirs(data_dir, exist_ok=True)
    # Make workload folders inside data folder
    os.makedirs(f'{data_dir}', exist_ok=True)
    os.makedirs(f'{data_dir}/times', exist_ok=True)
    os.makedirs(f'{data_dir}/docker', exist_ok=True)
    os.makedirs(f'{data_dir}/docker_processed', exist_ok=True)
    os.makedirs(f'{data_dir}/pcm', exist_ok=True)
    os.makedirs(f'{data_dir}/pcm_processed', exist_ok=True)
    return True

def move_files(data_dir):
    # Logs are files in data_dir with .log extension
    logs = [f for f in os.listdir('.') if f.endswith('.log')]
    for log in logs:
        params = log.split('_')
        log_type = params[0]
        # Move file to the appropriate folder
        os.rename(f'./{log}', f'{data_dir}/{log_type}/{log}')
    csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
    for csv in csvs:
        params = csv.split('_')
        csv_type = params[0]
        # Move file to the appropriate folder
        os.rename(f'./{csv}', f'{data_dir}/{csv_type}/{csv}')
    return True

def process_pcm(log_file, file_to_write):
    df2 = pd.read_csv(f'{faiss_results_dir}/pcm/{log_file}', skiprows=[0,1,2], iterator=True, chunksize=1000) #Dataframe starting at second row (necessary since df1 has no time column)
    
    first_chunk = True
    
    for chunk in df2:
        chunk_processed = chunk[['Date', 'Time', 'Proc Energy (Joules)', 'DRAM Energy (Joules)']]
        chunk_processed['Datetime'] = pd.to_datetime(chunk_processed['Date'] + ' ' + chunk_processed['Time'])
        chunk_processed = chunk_processed.drop(['Date', 'Time'], axis=1)
        position = chunk_processed.columns.get_loc('Datetime')
        chunk_processed['Time elapsed(s)'] =  (chunk_processed.iloc[1:, position] - chunk_processed.iat[0, position]).dt.total_seconds()
        chunk_processed = chunk_processed.fillna(0)

        chunk_processed['Socket 0 Proc Energy (Joules)'] = chunk['SKT0']
        chunk_processed['Socket 1 Proc Energy (Joules)'] = chunk['SKT1']
        chunk_processed['Socket 0 DRAM Energy (Joules)'] = chunk['SKT0.1']
        chunk_processed['Socket 1 DRAM Energy (Joules)'] = chunk['SKT1.1']

        chunk_processed['CPU Utilization (%)'] = 0

        # Sum CPU utilization for each socket
        num_cores = 96
        for core in range(num_cores):
            column = f'C0res%.{core + 6}'
            chunk_processed['CPU Utilization (%)'] += chunk[column].astype(float)

        chunk_processed['Socket 0 L3 Cache Occupancy'] = chunk['L3OCC']
        chunk_processed['Socket 1 L3 Cache Occupancy'] = chunk['L3OCC.1']

        chunk_processed['Memory Read Bandwidth (Socket 0) (GB/s)'] = chunk['READ.1']
        chunk_processed['Memory Write Bandwidth (Socket 0) (GB/s)'] = chunk['WRITE.1']
        chunk_processed['Memory Utilization (Socket 0) (%)'] = chunk['LOCAL.1']

        chunk_processed['Memory Read Bandwidth (Socket 1) (GB/s)'] = chunk['READ.2']
        chunk_processed['Memory Write Bandwidth (Socket 1) (GB/s)'] = chunk['WRITE.2']
        chunk_processed['Memory Utilization (Socket 1) (%)'] = chunk['LOCAL.2']

        chunk_processed['Memory Read Bandwidth (System) (GB/s)'] = chunk['READ']
        chunk_processed['Memory Write Bandwidth (System) (GB/s)'] = chunk['WRITE']
        chunk_processed['Memory Utilization (System) (%)'] = chunk['LOCAL']

        if first_chunk:
            chunk_processed.to_csv(file_to_write, index=False, header=True)
            first_chunk = False
        else:
            chunk_processed.to_csv(file_to_write, mode='a', index=False, header=False)

def process_docker_logs(data_dir):
    logs = os.listdir(f'{data_dir}/docker')

    for log in logs:
        csv = {
            'time': [],
            'cpu': [],
            'mem_usage': [],
            'mem_limit': [],
            'mem_%': [],
            'net_i/o_used': [],
            'net_i/o_total': [],
            'block_i/o_used': [],
            'block_i/o_total': []
        }
        time = ''
        with open(f'{data_dir}/docker/{log}') as file:
            print('Processing:', log)
            for line in file:
                # Check if the line starts with date time
                if re.search(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}$', line) != None:
                    time = line
                elif 'CONTAINER ID' in line:
                    continue
                else:
                    parts = line.split(' ')
                    parts = [part for part in parts if part != '']
                    if len(parts) < 13:
                        continue
                    else:
                        csv['cpu'].append(parts[2])
                        csv['mem_usage'].append(parts[3])
                        csv['mem_limit'].append(parts[5])
                        csv['mem_%'].append(parts[6])
                        csv['net_i/o_used'].append(parts[7])
                        csv['net_i/o_total'].append(parts[9])
                        csv['block_i/o_used'].append(parts[10])
                        csv['block_i/o_total'].append(parts[12])
                        csv['time'].append(time)

            df = pd.DataFrame(csv)
            df = df.fillna(0)
            # Convert memory all to GiB if it is in MiB
            for index, row in df.iterrows():
                if 'MiB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('MiB')[0]) / 1024
                elif 'MB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('MB')[0]) / 1024
                elif 'GiB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('GiB')[0])
                elif 'GB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('GB')[0])
                elif 'KiB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('KiB')[0]) / 1024 / 1024
                elif 'KB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('KB')[0]) / 1024 / 1024
                elif 'B' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('B')[0]) / 1024 / 1024 / 1024
                if 'MiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('MiB')[0]) / 1024
                elif 'MB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('MB')[0]) / 1024
                elif 'GiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('GiB')[0])
                elif 'GB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('GB')[0])
                elif 'KiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('KiB')[0]) / 1024 / 1024
                elif 'KB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('KB')[0]) / 1024 / 1024
                elif 'B' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('B')[0]) / 1024 / 1024 / 1024
            df.to_csv(f'{data_dir}/docker_processed/{log[:-4]}_processed.csv')

def process_faiss_logs(data_dir, batch_size_list):
    faiss_logs = os.listdir(f'{data_dir}/times')
    print('data dir:', data_dir)
    # Create a dataframe to store the data
    csv = {
        'index': [],
        'cpu': [],
        'batch size': [],
        'start time': [],
        'end time': [],
        'avg latency (ms)': [],
        '99th percentile (ms)': [],
        '95th percentile (ms)': [],
        'num queries': [],
    }
    for log in faiss_logs:
        # Load the log file in a dataframe
        df = pd.read_csv(f'{data_dir}/times/{log}')
        # Extract the index and cpu from the log file name
        index = log.split('_')[2].split('.')[0]
        cpu = int(log.split('_')[1])
        # Iterate over the batch sizes
        for batch_size in batch_size_list:
            # Filter the dataframe by the batch size
            df_batch = df[df['batch_size'] == batch_size]
            # For query time, split the string and convert to datetime
            df_batch['query_time'] = df_batch['query_time'].apply(lambda x: x.split(' ')[2])
            # Convert query time in datetime format to miliseconds
            df_batch['query_time'] = df_batch['query_time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S.%f').microsecond / 1000 + datetime.datetime.strptime(x, '%H:%M:%S.%f').second * 1000)
            # Calculate the average latency
            avg_latency = df_batch['query_time'].mean()
            # Calculate the 99th percentile
            percentile_99 = df_batch['query_time'].quantile(0.99)
            # Calculate the 95th percentile
            percentile_95 = df_batch['query_time'].quantile(0.95)
            # Calculate the number of queries
            num_queries = df_batch.shape[0] * batch_size
            # Find the start time
            start_time = df_batch['query_start'].min()
            # Find the end time
            end_time = df_batch['query_end'].max()
            # Add the data to the dataframe
            csv['index'].append(index)
            csv['cpu'].append(cpu)
            csv['batch size'].append(batch_size)
            csv['start time'].append(start_time)
            csv['end time'].append(end_time)
            csv['avg latency (ms)'].append(avg_latency)
            csv['99th percentile (ms)'].append(percentile_99)
            csv['95th percentile (ms)'].append(percentile_95)
            csv['num queries'].append(num_queries)

    # Create a dataframe from the dictionary
    df = pd.DataFrame(csv)

    # Sort the dataframe by index, cpu, and batch size
    df = df.sort_values(by=['index', 'cpu', 'batch size'])

    # Save the dataframe to a csv file
    df.to_csv(f'{data_dir}/faiss.csv')

def process_faiss_energy(data_dir, cpu_idle_power, dram_idle_power):
    # Create a dataframe to store the data
    df = pd.read_csv(f'{data_dir}/faiss.csv')
    print(df)
    pcm_logs = os.listdir(f'{data_dir}/pcm_processed')

    for pcm_log in pcm_logs:
        # Load the PCM log in a dataframe
        df_pcm = pd.read_csv(f'{data_dir}/pcm_processed/{pcm_log}')
        # Extract the index and cpu from the log file name
        cpu = int(pcm_log.split('_')[1])
        # Find the row in the dataframe that corresponds to the index and cpu
        df_cpu = df[df['cpu'] == cpu]
        # For each row in the dataframe, find the corresponding energy data in the PCM log
        for index, row in df_cpu.iterrows():
            # Filter the PCM log between the start and end time
            df_pcm_cpu = df_pcm[(df_pcm['Datetime'] >= row['start time']) & (df_pcm['Datetime'] <= row['end time'])]
            # Find the energy consumed by the CPU
            energy_cpu = df_pcm_cpu['Proc Energy (Joules)'].sum()
            # Find the energy consumed by the DRAM
            energy_dram = df_pcm_cpu['DRAM Energy (Joules)'].sum()
            start_time = datetime.datetime.strptime(row['start time'], '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.datetime.strptime(row['end time'], '%Y-%m-%d %H:%M:%S.%f')
            runtime = (end_time - start_time).total_seconds()
            # Find the idle energy consumed by the CPU
            cpu_idle_energy = cpu_idle_power * runtime
            # Find the idle energy consumed by the DRAM
            dram_idle_energy = dram_idle_power * runtime
            # Find the dynamic energy consumed by the CPU
            cpu_dynamic_energy = energy_cpu - cpu_idle_energy
            # Find the dynamic energy consumed by the DRAM
            dram_dynamic_energy = energy_dram - dram_idle_energy
            # Find the average CPU utilization
            df.at[index, 'Average CPU Utilization (%)'] = df_pcm_cpu['CPU Utilization (%)'].mean()
            # Add the energy data to the dataframe
            df.at[index, 'Runtime (s)'] = runtime
            df.at[index, 'Average CPU Power (W)'] = energy_cpu / runtime
            df.at[index, 'Average DRAM Power (W)'] = energy_dram / runtime
            print('File:', pcm_log)
            print('Average DRAM Power (W):', energy_dram / runtime)
            df.at[index, 'Total CPU Energy (Joules)'] = energy_cpu
            df.at[index, 'Total CPU Idle Energy (Joules)'] = cpu_idle_energy
            df.at[index, 'Total CPU Dynamic Energy (Joules)'] = cpu_dynamic_energy
            df.at[index, 'Total DRAM Energy (Joules)'] = energy_dram
            df.at[index, 'Total DRAM Idle Energy (Joules)'] = dram_idle_energy
            df.at[index, 'Total DRAM Dynamic Energy (Joules)'] = dram_dynamic_energy
            df.at[index, 'Total Energy (Joules)'] = energy_cpu + energy_dram
            df.at[index, 'Total Idle Energy (Joules)'] = cpu_idle_energy + dram_idle_energy
            df.at[index, 'Total Dynamic Energy (Joules)'] = cpu_dynamic_energy + dram_dynamic_energy
            # Find the energy consumed by the CPU per query
            df.at[index, 'CPU Energy per Query (Joules)'] = energy_cpu / row['num queries']
            df.at[index, 'CPU Idle Energy per Query (Joules) Unscaled'] = cpu_idle_energy / row['num queries']
            df.at[index, 'CPU Dynamic Energy per Query (Joules)'] = cpu_dynamic_energy / row['num queries']
            # Find the energy consumed by the DRAM per query
            df.at[index, 'DRAM Energy per Query (Joules)'] = energy_dram / row['num queries']
            df.at[index, 'DRAM Idle Energy per Query (Joules) Unscaled'] = dram_idle_energy / row['num queries']
            df.at[index, 'DRAM Dynamic Energy per Query (Joules)'] = dram_dynamic_energy / row['num queries']
            # Find the total energy consumed per query
            df.at[index, 'Total Energy per Query (Joules)'] = (energy_cpu + energy_dram) / row['num queries']
            df.at[index, 'Total Idle Energy per Query (Joules) Unscaled'] = (cpu_idle_energy + dram_idle_energy) / row['num queries']
            df.at[index, 'Total Dynamic Energy per Query (Joules)'] = (cpu_dynamic_energy + dram_dynamic_energy) / row['num queries']

    # Rename lower case columns
    df = df.rename(columns={'index': 'Index', 'cpu': 'CPU Cores', 'batch size': 'Batch Size', 'avg latency (ms)': 'Average Latency (ms)', '99th percentile (ms)': '99th Percentile (ms)', '95th percentile (ms)': '95th Percentile (ms)', 'num queries': 'Number of Queries'})
    # Find throughput
    df['Throughput (queries/s)'] = df['Number of Queries'] / df['Runtime (s)']
    # Find resource usage
    df['CPU-Seconds Per Query'] = df['CPU Cores'] * df['Runtime (s)'] / df['Number of Queries']
    # Drop unamed column
    df = df.drop(['Unnamed: 0'], axis=1)
    # Throw an error if the dataframe has any negative energy values
    if (df['Total Dynamic Energy per Query (Joules)'] < 0).any():
        raise ValueError('Negative total dynamic energy values found in the dataframe')
    if (df['CPU Dynamic Energy per Query (Joules)'] < 0).any():
        raise ValueError('Negative CPU dynamic energy values found in the dataframe')
    if (df['DRAM Dynamic Energy per Query (Joules)'] < 0).any():
        raise ValueError('Negative DRAM dynamic energy values found in the dataframe')
    # Save the dataframe to a csv file
    df.to_csv(f'{data_dir}/faiss_processed.csv')

def find_memory_footprint(data_dir, cpu_per_node, memory_per_node):
    docker_logs = os.listdir(f'{data_dir}/docker_processed')
    faiss_df = pd.read_csv(f'{data_dir}/faiss_processed.csv')
    for docker_log in docker_logs:
        cpu = int(docker_log.split('_')[1])
        # Load the Docker log in a dataframe
        df_docker = pd.read_csv(f'{data_dir}/docker_processed/{docker_log}')
        faiss_df_cpu = faiss_df[faiss_df['CPU Cores'] == cpu]
        for index, row in faiss_df_cpu.iterrows():
            # Filter the Docker log between the start and end time
            df_docker_cpu = df_docker[(df_docker['time'] >= row['start time']) & (df_docker['time'] <= row['end time'])]
            # Find the memory usage
            # Convert memory usage to float
            df_docker_cpu['mem_usage'] = df_docker_cpu['mem_usage'].astype(float)
            memory_usage = df_docker_cpu['mem_usage'].quantile(0.99)
            # Round the memory usage to nearest integer
            memory_usage = round(memory_usage)
            memory_seconds_per_query = memory_usage * row['Runtime (s)'] / row['Number of Queries']
            # Add the memory footprint to the dataframe
            faiss_df.at[index, 'Memory Footprint (GB)'] = memory_usage
            faiss_df.at[index, 'Memory Usage per Query (GB-second)'] = memory_seconds_per_query
            faiss_df.at[index, 'CPU Idle Energy per Query (Joules)'] = row['CPU Idle Energy per Query (Joules) Unscaled'] * cpu / cpu_per_node
            faiss_df.at[index, 'CPU Energy per Query (Joules)'] = faiss_df.at[index, 'CPU Idle Energy per Query (Joules)'] + row['CPU Dynamic Energy per Query (Joules)']
            faiss_df.at[index, 'DRAM Idle Energy per Query (Joules)'] = row['DRAM Idle Energy per Query (Joules) Unscaled'] * memory_usage / memory_per_node
            faiss_df.at[index, 'DRAM Energy per Query (Joules)'] = faiss_df.at[index, 'DRAM Idle Energy per Query (Joules)'] + row['DRAM Dynamic Energy per Query (Joules)']
            faiss_df.at[index, 'Total Idle Energy per Query (Joules)'] = faiss_df.at[index, 'CPU Idle Energy per Query (Joules)'] + faiss_df.at[index, 'DRAM Idle Energy per Query (Joules)']
            faiss_df.at[index, 'Total Energy per Query (Joules)'] = faiss_df.at[index, 'CPU Energy per Query (Joules)'] + faiss_df.at[index, 'DRAM Energy per Query (Joules)']
    # Save the dataframe to a csv file
    faiss_df.to_csv(f'{data_dir}/faiss_processed.csv')

def find_carbon_footprint(grid_ci, cpu_ci, mem_ci, in_faiss_df):
    faiss_df = in_faiss_df.copy()
    faiss_df['CPU Embodied CI (gCO2eq/CPU-s)'] = cpu_ci
    faiss_df['Memory Embodied CI (gCO2eq/GB-s)'] = mem_ci
    # Operational carbon footprint
    faiss_df[f'CPU Operational Carbon per Query (gCO2eq)'] = faiss_df['CPU Energy per Query (Joules)'] * grid_ci
    faiss_df[f'Memory Operational Carbon per Query (gCO2eq)'] = faiss_df['DRAM Energy per Query (Joules)'] * grid_ci
    faiss_df[f'Total Operational Carbon per Query (gCO2eq)'] = faiss_df[f'CPU Operational Carbon per Query (gCO2eq)'] + faiss_df[f'Memory Operational Carbon per Query (gCO2eq)']
    # Create a new column for embodied carbon footprint
    faiss_df[f'CPU Embodied Carbon per Query (gCO2eq)'] = faiss_df['CPU-Seconds Per Query'] * cpu_ci
    faiss_df[f'Memory Embodied Carbon per Query (gCO2eq)'] = faiss_df['Memory Usage per Query (GB-second)'] * mem_ci
    faiss_df[f'Total Embodied Carbon per Query (gCO2eq)'] = faiss_df[f'CPU Embodied Carbon per Query (gCO2eq)'] + faiss_df[f'Memory Embodied Carbon per Query (gCO2eq)']
    # Create a new column for total carbon footprint
    faiss_df[f'Total Carbon Footprint per Query (gCO2eq)'] = faiss_df[f'Total Operational Carbon per Query (gCO2eq)'] + faiss_df[f'Total Embodied Carbon per Query (gCO2eq)']
    return faiss_df

def co2(faiss_df, cpu_ci, mem_ci, grid_ci_list):
    for grid_ci in grid_ci_list:
        kWh_to_J = 3.6e6
        co2_df = find_carbon_footprint(grid_ci / kWh_to_J, cpu_ci, mem_ci, faiss_df)
        # Add the grid carbon intensity as a column
        co2_df['Grid CI (gCO2eq/kWh)'] = grid_ci
        co2_df.to_csv(f'{data_dir}/co2_faiss_data_{grid_ci}.csv')

def total_carbon_analysis(grid_ci_list, cpu_ci, mem_ci):
    # Read the combined data
    faiss_df = pd.read_csv(f'{data_dir}/faiss_processed.csv')
    co2(faiss_df, cpu_ci, mem_ci, grid_ci_list)

def latency_99_carbon_pareto_front(grid_ci_list, pareto_threshold):
    # For each workload + index + grid ci, plot latency_99 vs. carbon footprint but only keep the points on the pareto front
    df = pd.concat([pd.read_csv(f'{data_dir}/co2_faiss_data_{grid_ci}.csv') for grid_ci in grid_ci_list])
    for grid_ci in grid_ci_list:
        plt.figure(figsize=(15,10))
        grid_ci_df = df[df['Grid CI (gCO2eq/kWh)'] == grid_ci]
        # Sort by batch_size
        grid_ci_df['pareto'] = False
        for index, row in grid_ci_df.iterrows():
            pareto = True
            for i, x in grid_ci_df.iterrows():
                if row['99th Percentile (ms)'] > (pareto_threshold * x['99th Percentile (ms)']) and row['Total Carbon Footprint per Query (gCO2eq)'] > (x['Total Carbon Footprint per Query (gCO2eq)']) and index != i:
                    pareto = False
                    break
                elif row['99th Percentile (ms)'] > (x['99th Percentile (ms)']) and row['Total Carbon Footprint per Query (gCO2eq)'] > (pareto_threshold * x['Total Carbon Footprint per Query (gCO2eq)']) and index != i:
                    pareto = False
                    break
                else:
                    continue
            if pareto == True:
                grid_ci_df.loc[index, 'pareto'] = True
            else:
                grid_ci_df.loc[index, 'pareto'] = False
                            
        pareto_front_df = grid_ci_df[grid_ci_df['pareto'] == True]
        # Save the pareto front to a csv file
        pareto_front_df.to_csv(f'{data_dir}/pareto_front_99_{grid_ci}.csv')

def latency_99_carbon_pareto_front_1x2_grid(grid_ci_list, grid_ci_labels, pareto_threshold):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=False)
    for i, grid_ci in enumerate(grid_ci_list):
        pareto_front_df = pd.read_csv(f'{data_dir}/pareto_front_99_{grid_ci}.csv')
        # Sort pareto front by latency
        pareto_front_df = pareto_front_df.sort_values(by='99th Percentile (ms)')
        # Scatter plot all points, label points with cpu and batch_size
        for index, row in pareto_front_df.iterrows():
            if row['Index'] == 'hnsw':
                marker = 'o'
            elif row['Index'] == 'ivf':
                marker = 'x'
            axs[i].scatter(row['99th Percentile (ms)']/1000, row['Total Carbon Footprint per Query (gCO2eq)'] * 1000, label=f'{row["CPU Cores"], row["Batch Size"]}', marker=marker, s=150)
            # Label the first and last points with the cpu and batch_size
            # if index == pareto_front_df.index[0]:
            #     axs[i].text(row['Average Latency (ms)']/1000, row['Total Carbon Footprint per Query (gCO2eq)'] * 1000, f'{row["CPU Cores"]} CPUs, Batch Size = {row["Batch Size"]}', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
            # elif index == pareto_front_df.index[-1]:
            #     axs[i].text(row['Average Latency (ms)']/1000, row['Total Carbon Footprint per Query (gCO2eq)'] * 1000, f'{row["CPU Cores"]} CPUs, Batch Size = {row["Batch Size"]}', fontsize=12, verticalalignment='top', horizontalalignment='right')
        # 2 column legend
        axs[i].legend(fontsize=13, ncol=2, title='CPU Cores, Batch Size', title_fontsize=14)
        axs[i].tick_params(axis='x', labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
        axs[i].set_title(f'{grid_ci_labels[i]} - {grid_ci} gCO2eq/kWh', fontsize=18)

    fig.text(0.5, 0.02, '99th Percentile Tail Latency (s)', ha='center', fontsize=20)
    # Label the y-axis centered in the whole plot
    fig.text(0.01, 0.5, 'Carbon ($CO_{2}eq/query$)', va='center', rotation='vertical', fontsize=20)
    # Legend for the whole plot on top center, items horizontally aligned for the line types, x for ivf, o for hnsw
    legend_handles = [mlines.Line2D([0], [0], marker='x', color='w', label='IVF', markerfacecolor='black', markeredgecolor='black', markersize=15), mlines.Line2D([0], [0], marker='o', color='w', label='HNSW', markerfacecolor='black', markeredgecolor='black', markersize=15)]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.81)
    plt.savefig(f'{fair_co2_path}/figures/12_faiss_pareto_front_latency_99_vs_cf_1x2_grid.png', dpi=300)
    # plt.show()
    plt.close()

if __name__ == '__main__':

    data_dir = faiss_results_dir

    interval = 0.1

    lifetime = 4 * 365 * 24 * 60 * 60# seconds
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

    # Run the scripts in order
    create_folders(data_dir)
    move_files(data_dir)
    print('Data moved successfully')

    # Process PCM    
    pcm_logs = os.listdir(f'{faiss_results_dir}/pcm')
    for pcm_log in pcm_logs:
        raw_filename = pcm_log[:pcm_log.find('.log')]
        if (f'{raw_filename}_processed.csv' in os.listdir(f'{faiss_results_dir}/pcm_processed')):
            print('Skipping', pcm_log)
            continue
        else:
            print('Processing', pcm_log)
            process_pcm(pcm_log, f'{faiss_results_dir}/pcm_processed/{raw_filename}_processed.csv')
    print('PCM logs processed successfully')
    
    # # Process docker stats
    # process_docker_logs(data_dir)
    # print('Docker stats processed successfully')

    # # Process FAISS times
    # batch_size_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    # process_faiss_logs(data_dir, batch_size_list)
    # process_faiss_energy(data_dir, idle_cpu_power, idle_dram_power)
    # find_memory_footprint(data_dir, num_cores_per_cpu * num_cpus_per_node, gb_per_node)

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

    kWh_per_joule = 1 / 3600000

    grid_ci_list = [41, 369]
    grid_ci_labels = ['Sweden', 'US']
    total_carbon_analysis(grid_ci_list, cpu_ci, mem_ci)
    pareto_threshold = 0.999
    latency_99_carbon_pareto_front(grid_ci_list, pareto_threshold)
    latency_99_carbon_pareto_front_1x2_grid(grid_ci_list, grid_ci_labels, pareto_threshold)
    print('Carbon analysis completed successfully')
    print('All scripts executed successfully')