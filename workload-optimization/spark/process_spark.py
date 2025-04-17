# Main script that runs all the processing scripts
import os
import pandas as pd
import re

fair_co2_path = os.environ.get('FAIR_CO2')
spark_results_dir = f'{fair_co2_path}/workload-optimization/results'

def create_folders(workload_list):
    # Make data folder
    os.makedirs(spark_results_dir, exist_ok=True)
    # Make workload folders inside data folder
    for workload in workload_list:
        os.makedirs(f'{spark_results_dir}/{workload}', exist_ok=True)
        os.makedirs(f'{spark_results_dir}/{workload}/logs', exist_ok=True)
        os.makedirs(f'{spark_results_dir}/{workload}/docker', exist_ok=True)
        os.makedirs(f'{spark_results_dir}/{workload}/docker_processed', exist_ok=True)
        os.makedirs(f'{spark_results_dir}/{workload}/pcm', exist_ok=True)
        os.makedirs(f'{spark_results_dir}/{workload}/pcm_processed', exist_ok=True)
    return True

def move_files(data_dir, workload_list):
    # Logs are files in data_dir with .log extension
    logs = [f for f in os.listdir('.') if f.endswith('.log')]
    for log in logs:
        params = log.split('_')
        log_type = params[0]
        workload = params[1]
        # Move file to the appropriate folder
        if workload in workload_list:
            os.rename(f'./{log}', f'{data_dir}/{workload}/{log_type}/{log}')
    return True

def process_pcm(log_file, workload, file_to_write):
    df2 = pd.read_csv(f'{spark_results_dir}/{workload}/pcm/{log_file}', skiprows=[0,1,2], iterator=True, chunksize=1000) #Dataframe starting at second row (necessary since df1 has no time column)
    
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

def process_spark_logs(data_dir, workload): 
    csv = {
        'memory' : [],
        'threads' : [],
        'read_file runtime' : [],
        'total sum column runtime' : [],
        'total count distinct runtime' : [],
        'total aggregate runtime' : [],
        'avg sum column runtime' : [],
        'avg count distinct runtime' : [],
        'avg aggregate runtime' : [],
        'rounds' : [],
        'start' : [],
        'end' : []
        #'Total Energy' : [],
        #'Average Energy' : []
    }

    logs = os.listdir(f'{data_dir}/{workload}/spark')
    for log in logs:
        parts = log.split('_')
        driver_mem = parts[1]
        threads = int(parts[2][:parts[2].find('.log')])

        with open(f'{data_dir}/{workload}/spark/{log}') as f:
            sum_count = 0
            distinct_count = 0
            aggregate_count = 0
            
            sum_time = 0
            distinct_time = 0
            aggregate_time = 0

            start = 0
            end = 0
            for line in f:
                if 'read file' in line and 'elapsed seconds' in line:
                    read_file = float(line[line.rfind(' ') + 1:])
                elif 'sum' in line:
                    if 'round 2 starting at' in line and 'sum2' not in line and 'sum3' not in line:
                        start = float(line[line.rfind(' ') + 1:])
                    elif ('round 2' in line) and 'elapsed seconds' in line:
                        sum_time += float(line[line.rfind(' ') + 1:])
                        sum_count += 1
                elif 'count distinct' in line:
                    if ('round 2' in line) and 'elapsed seconds' in line:
                        distinct_time += float(line[line.rfind(' ') + 1:])
                        distinct_count += 1
                elif 'aggregate' in line:
                    if 'round 2 ended at' in line and 'aggregate3' in line:
                        end = float(line[line.rfind(' ') + 1:])
                    elif ('round ' in line) and 'elapsed seconds' in line:
                        aggregate_time +=  float(line[line.rfind(' ') + 1:])
                        aggregate_count += 1
                    
            if sum_count == 0 or distinct_count == 0 or aggregate_count == 0:
                continue
            
            csv['memory'].append(int(driver_mem[:-1]))
            csv['threads'].append(threads)
            csv['read_file runtime'].append(read_file)

            
            csv['total sum column runtime'].append(sum_time)
            csv['total count distinct runtime'].append(distinct_time)
            csv['total aggregate runtime'].append(aggregate_time)

            csv['avg sum column runtime'].append(sum_time/sum_count)
            csv['avg count distinct runtime'].append(distinct_time/distinct_count)
            csv['avg aggregate runtime'].append(aggregate_time/aggregate_count)

            csv['rounds'].append(sum_count)

            csv['start'].append(start)
            csv['end'].append(end)
            
    df = pd.DataFrame(csv)
    df['total_runtime'] = df['total sum column runtime'] + df['total count distinct runtime'] + df['total aggregate runtime']
    df = df.sort_values(['memory','threads'])
    df.to_csv(f'{data_dir}/{workload}/spark.csv')

def process_docker_logs(data_dir, workload):
    logs = os.listdir(f'{data_dir}/{workload}/docker')

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
        with open(f'{data_dir}/{workload}/docker/{log}') as file:
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
                elif 'GiB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('GiB')[0])
                elif 'KiB' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('KiB')[0]) / 1024 / 1024
                elif 'B' in row['mem_usage']:
                    df.at[index, 'mem_usage'] = float(row['mem_usage'].split('B')[0]) / 1024 / 1024 / 1024
                if 'MiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('MiB')[0]) / 1024
                elif 'GiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('GiB')[0])
                elif 'KiB' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('KiB')[0]) / 1024 / 1024
                elif 'B' in row['mem_limit']:
                    df.at[index, 'mem_limit'] = float(row['mem_limit'].split('B')[0]) / 1024 / 1024 / 1024
            df.to_csv(f'{data_dir}/{workload}/docker_processed/{log[:-4]}_processed.csv')

def scale_pcm_power(data_dir, workload_list, interval):
    for workload in workload_list:
        for pcm_log in os.listdir(f'{data_dir}/{workload}/pcm_processed'):
            pcm_df = pd.read_csv(f'{data_dir}/{workload}/pcm_processed/{pcm_log}')
            pcm_df['Proc Power (Watts)'] = pcm_df['Proc Energy (Joules)'] / interval
            pcm_df['DRAM Power (Watts)'] = pcm_df['DRAM Energy (Joules)'] / interval
            pcm_df['Total Power (Watts)'] = pcm_df['Proc Power (Watts)'] + pcm_df['DRAM Power (Watts)']
            pcm_df.to_csv(f'{data_dir}/{workload}/pcm_processed/{pcm_log}')

def scaling_analysis(data_dir, workload, num_cpus_per_node, dram_per_node, cpu_idle_power, dram_idle_power):
    # Read in processed pbbs logs
    df = pd.read_csv(f'{data_dir}/{workload}/spark.csv')
    print(f'CPU Idle Power: {cpu_idle_power}')
    print(f'DRAM Idle Power: {dram_idle_power}')
    # Remove unamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # for each entry in the dataframe, retrieve the relevant pcm data file and plot the energy consumption
    for index, row in df.iterrows():
        threads = int(row['threads'])
        memory = int(row['memory'])
        runtime = row['total_runtime']
        pcm_file = f'{data_dir}/{workload}/pcm_processed/{workload}_{memory}g_{threads}_processed.csv'
        pcm_df = pd.read_csv(pcm_file)
        pcm_df = pcm_df.loc[:, ~pcm_df.columns.str.contains('^Unnamed')]
        pcm_df['Datetime'] = pd.to_datetime(pcm_df['Datetime'])
        if 'start' in list(df.columns) and 'end' in list(df.columns):
            pcm_df = pcm_df[(pcm_df['timestamp'] >= row['start']) & (pcm_df['timestamp'] <= row['end'])]
        average_cpu_power = pcm_df['Proc Power (Watts)'].mean()
        average_dram_power = pcm_df['DRAM Power (Watts)'].mean()
        average_cpu_energy_per_round = average_cpu_power * runtime
        average_dram_energy_per_round = average_dram_power * runtime
        average_cpu_idle_energy_per_round = cpu_idle_power * runtime
        average_dram_idle_energy_per_round = dram_idle_power * runtime
        average_cpu_dynamic_energy_per_round = average_cpu_energy_per_round - average_cpu_idle_energy_per_round
        average_dram_dynamic_energy_per_round = average_dram_energy_per_round - average_dram_idle_energy_per_round
        if average_cpu_dynamic_energy_per_round < 0:
            print(f'CPU Power: {average_cpu_power}')
            print(f'DRAM Power: {average_dram_power}')
            raise ValueError('CPU dynamic energy per round is negative')

        if average_dram_dynamic_energy_per_round < 0:
            raise ValueError('DRAM dynamic energy per round is negative')

        df.at[index, 'Average CPU Dynamic Energy per Round'] = average_cpu_dynamic_energy_per_round
        df.at[index, 'Average DRAM Dynamic Energy per Round'] = average_dram_dynamic_energy_per_round
        df.at[index, 'Average CPU Energy per Round Unscaled'] = average_cpu_energy_per_round
        df.at[index, 'Average DRAM Energy per Round Unscaled'] = average_dram_energy_per_round
        df.at[index, 'Average CPU Energy per Round'] = average_cpu_dynamic_energy_per_round + threads/num_cpus_per_node * average_cpu_idle_energy_per_round
        df.at[index, 'Average DRAM Energy per Round'] = average_dram_dynamic_energy_per_round + memory/dram_per_node * average_dram_idle_energy_per_round

        # Add average power consumption to dataframe
        df.at[index, 'Average CPU Power'] = average_cpu_power
        df.at[index, 'Average DRAM Power'] = average_dram_power

        # Add CPU-seconds to dataframe
        df.at[index, 'CPU-seconds'] = threads * runtime

        # Add memory-seconds to dataframe
        df.at[index, 'Memory-seconds'] = memory * runtime

        # Save the updated dataframe
        df.to_csv(f'{data_dir}/{workload}/spark.csv')

def find_carbon_footprint(grid_ci, cpu_ci, mem_ci, in_pbbs_df):
    pbbs_df = in_pbbs_df.copy()
    pbbs_df['CPU Embodied CI (gCO2eq/CPU-s)'] = cpu_ci
    pbbs_df['Memory Embodied CI (gCO2eq/GiB-s)'] = mem_ci
    # Operational carbon footprint
    pbbs_df[f'CPU Operational Carbon'] = pbbs_df['Average CPU Energy per Round'] * grid_ci
    pbbs_df[f'Memory Operational Carbon'] = pbbs_df['Average DRAM Energy per Round'] * grid_ci
    pbbs_df[f'Total Operational Carbon'] = pbbs_df[f'CPU Operational Carbon'] + pbbs_df[f'Memory Operational Carbon']
    # Create a new column for embodied carbon footprint
    pbbs_df[f'CPU Embodied Carbon'] = pbbs_df['CPU-seconds'] * cpu_ci
    pbbs_df[f'Memory Embodied Carbon'] = pbbs_df['Memory-seconds'] * mem_ci
    pbbs_df[f'Total Embodied Carbon'] = pbbs_df[f'CPU Embodied Carbon'] + pbbs_df[f'Memory Embodied Carbon']
    # Create a new column for total carbon footprint
    pbbs_df[f'Total Carbon Footprint'] = pbbs_df[f'Total Operational Carbon'] + pbbs_df[f'Total Embodied Carbon']
    return pbbs_df

def co2(pbbs_df, cpu_ci, mem_ci, grid_ci_list, workload):
    for grid_ci in grid_ci_list:
        kWh_to_J = 3.6e6
        co2_df = find_carbon_footprint(grid_ci / kWh_to_J, cpu_ci, mem_ci, pbbs_df)
        # Add the grid carbon intensity as a column
        co2_df['workload'] = workload
        co2_df['Grid CI (gCO2eq/kWh)'] = grid_ci
        co2_df.to_csv(f'{spark_results_dir}/{workload}/co2_{workload}_data_{grid_ci}.csv')

def total_carbon_analysis(workload_list, grid_ci_list, cpu_ci, mem_ci):
    # Read the combined data
    for workload in workload_list:
        spark_df = pd.read_csv(f'{spark_results_dir}/{workload}/spark.csv')
        co2(spark_df, cpu_ci, mem_ci, grid_ci_list, workload)

def carbon_comparison(workload_list, grid_ci_list):
    # For each workload, find the performance optimal configuration
    for workload in workload_list:
        workload_df = pd.concat([pd.read_csv(f'{spark_results_dir}/{workload}/co2_{workload}_data_{grid_ci}.csv') for grid_ci in grid_ci_list])
        workload_df['Min CF'] = False
        workload_df['Min Runtime'] = False
        workload_df['Min Energy'] = False
        workload_df['Min Embodied'] = False
        for grid_ci in grid_ci_list:
            grid_ci_df = workload_df[workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci]
            # Find the minimum carbon footprint
            min_cf = workload_df['Total Carbon Footprint'].min()
            min_cpu = workload_df[workload_df['Total Carbon Footprint'] == min_cf]['threads'].values[0]
            min_memory = workload_df[workload_df['Total Carbon Footprint'] == min_cf]['memory'].values[0]
            # Label the row with the minimum carbon footprint in the combined dataframe
            workload_df.loc[(workload_df['workload'] == workload) & (workload_df['threads'] == min_cpu) & (workload_df['memory'] == min_memory) & (workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci), 'Min CF'] = True
            min_runtime = workload_df['total_runtime'].min()
            fastest_cpu = workload_df[workload_df['total_runtime'] == min_runtime]['threads'].values[0]
            fastest_memory = workload_df[workload_df['total_runtime'] == min_runtime]['memory'].values[0]
            # Label the row with the minimum runtime in the combined dataframe
            workload_df.loc[(workload_df['workload'] == workload) & (workload_df['threads'] == fastest_cpu) & (workload_df['memory'] == fastest_memory) & (workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci), 'Min Runtime'] = True
            # Find the minimum energy
            min_energy = workload_df['Total Operational Carbon'].min()
            min_cpu = workload_df[workload_df['Total Operational Carbon'] == min_energy]['threads'].values[0]
            min_memory = workload_df[workload_df['Total Operational Carbon'] == min_energy]['memory'].values[0]  
            # Label the row with the minimum energy in the combined dataframe
            workload_df.loc[(workload_df['workload'] == workload) & (workload_df['threads'] == min_cpu) & (workload_df['memory'] == min_memory) & (workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci), 'Min Energy'] = True
            # Find the minimum embodied carbon
            min_embodied = workload_df['Total Embodied Carbon'].min()
            min_cpu = workload_df[workload_df['Total Embodied Carbon'] == min_embodied]['threads'].values[0]
            min_memory = workload_df[workload_df['Total Embodied Carbon'] == min_embodied]['memory'].values[0]
            # Label the row with the minimum embodied carbon in the combined dataframe
            workload_df.loc[(workload_df['workload'] == workload) & (workload_df['threads'] == min_cpu) & (workload_df['memory'] == min_memory) & (workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci), 'Min Embodied'] = True
        # Drop unamed columns
        workload_df = workload_df.drop(columns=['Unnamed: 0'])
        workload_df = workload_df.drop(columns=['Unnamed: 0.1'])
        # Save the labeled dataframe
        workload_df.to_csv(f'{spark_results_dir}/{workload}/combined_co2_{workload}_data.csv')

def filter_co2_data(workload_list, grid_ci_list):
    # For each workload + input, calculate the difference in carbon footprint between the different configurations
    for workload in workload_list:
        workload_df = pd.read_csv(f'{spark_results_dir}/{workload}/combined_co2_{workload}_data.csv')
        filtered_co2_df = workload_df[(workload_df['Min CF'] == True) | (workload_df['Min Runtime'] == True) | (workload_df['Min Energy'] == True) | (workload_df['Min Embodied'] == True)]
        for grid_ci in grid_ci_list:
            grid_ci_df = workload_df[workload_df['Grid CI (gCO2eq/kWh)'] == grid_ci]
            min_cf = workload_df[workload_df['Min CF'] == True]['Total Carbon Footprint'].values[0]
            min_runtime = workload_df[workload_df['Min Runtime'] == True]['Total Carbon Footprint'].values[0]
            min_energy = workload_df[workload_df['Min Energy'] == True]['Total Carbon Footprint'].values[0]
            min_embodied = workload_df[workload_df['Min Embodied'] == True]['Total Carbon Footprint'].values[0]
            for index, row in workload_df.iterrows():
                cf = row['Total Carbon Footprint']
                cf_runtime_diff = (row['Total Carbon Footprint'] - min_runtime) / min_runtime * 100
                cf_energy_diff = (row['Total Carbon Footprint'] - min_energy) / min_energy * 100
                cf_embodied_diff = (row['Total Carbon Footprint'] - min_embodied) / min_embodied * 100
                cf_min_diff = (row['Total Carbon Footprint'] - min_cf) / min_cf * 100
                # Add the differences to the dataframe
                filtered_co2_df.loc[filtered_co2_df['Total Carbon Footprint'] == cf, 'CF Improvement over Perf-Optimal (%)'] = -cf_runtime_diff
                filtered_co2_df.loc[filtered_co2_df['Total Carbon Footprint'] == cf, 'CF Improvement over Energy-Optimal (%)'] = -cf_energy_diff
                filtered_co2_df.loc[filtered_co2_df['Total Carbon Footprint'] == cf, 'CF Improvement over Embodied-Optimal (%)'] = -cf_embodied_diff
                filtered_co2_df.loc[filtered_co2_df['Total Carbon Footprint'] == cf, 'CF Improvement over CF-Optimal (%)'] = -cf_min_diff
        filtered_co2_df_min_cf_only = filtered_co2_df[filtered_co2_df['Min CF'] == True]
        # Drop unamed columns
        filtered_co2_df = filtered_co2_df.drop(columns=['Unnamed: 0'])
        filtered_co2_df_min_cf_only = filtered_co2_df_min_cf_only.drop(columns=['Unnamed: 0'])
        # Save the filtered dataframe
        filtered_co2_df.to_csv(f'{spark_results_dir}/{workload}/filtered_co2_{workload}_data.csv')
        filtered_co2_df_min_cf_only.to_csv(f'{spark_results_dir}/{workload}/filtered_co2_{workload}_data_min_cf_only.csv')
    return filtered_co2_df, filtered_co2_df_min_cf_only

if __name__ == '__main__':
    workload_list = [
        'spark'
    ]   
    data_dir = spark_results_dir
    interval = 1
    
    # Run the scripts in order
    create_folders(workload_list)
    move_files(data_dir, workload_list)
    print('Data moved successfully')
    # Process PCM    
    for workload in workload_list:
        pcm_logs = os.listdir(f'{spark_results_dir}/{workload}/pcm')
        for pcm_log in pcm_logs:
            raw_filename = pcm_log[:pcm_log.find('.log')]
            if (f'{raw_filename}_processed.csv' in os.listdir(f'{spark_results_dir}/{workload}/pcm_processed')):
                print('Skipping', pcm_log)
                continue
            else:
                print('Processing', pcm_log)
                process_pcm(pcm_log, workload, f'{spark_results_dir}/{workload}/pcm_processed/{raw_filename}_processed.csv')
    print('PCM logs processed successfully')
    
    # Process spark logs
    process_spark_logs(data_dir, workload)
    print('Spark logs processed successfully')

    # # Process docker stats
    # for workload in workload_list:
    #     process_docker_logs(data_dir, workload)
    # print('Docker stats processed successfully')

    # Carbon analysis
    # Define the carbon intensity values for CPU and memory
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
    cpu_chip_cf_per_cpu = imec_cpu_chip_cf
    cpu_cf = cpu_chip_cf_per_cpu * num_cpus_per_node # gCO2eq
    cpu_cf_per_core = cpu_cf / num_cores_per_cpu # gCO2eq
    
    cpu_and_cooling_cf = cpu_cf + cooling_cf # gCO2eq
    cpu_and_cooling_cf_per_core = cpu_and_cooling_cf / num_cores_per_cpu # gCO2eq

    dram_cf_per_gb =  dram_cf / gb_per_node# gCO2eq

    other_cf = mb_cf + chassis_cf + peripheral_cf + psu_cf + ssd_cf # gCO2eq
    node_cf = (cpu_and_cooling_cf + dram_cf + ssd_cf + mb_cf + chassis_cf + peripheral_cf + psu_cf)

    mem_ci = (dram_cf + 0.5 * other_cf) / gb_per_node / lifetime # gCO2eq/(GB-second)
    cpu_ci = (cpu_and_cooling_cf + 0.5 * other_cf) / (num_cores_per_cpu * num_cpus_per_node) / lifetime # gCO2eq/(core-second)
    node_ci = node_cf / lifetime # gCO2eq/node-second

    scale_pcm_power(data_dir, workload_list, interval)
    # Process and plot data
    for workload in workload_list:
        scaling_analysis(data_dir, workload, num_cpus_per_node=num_cores_per_cpu * num_cpus_per_node, dram_per_node=gb_per_node, cpu_idle_power=idle_cpu_power, dram_idle_power=idle_dram_power)

    grid_ci_list = range (0, 410, 10) # gCO2eq/kWh
    total_carbon_analysis(workload_list, grid_ci_list, cpu_ci, mem_ci)
    carbon_comparison(workload_list, grid_ci_list)
    filter_co2_data(workload_list, grid_ci_list)
    print('Carbon analysis completed successfully')
    print('All scripts executed successfully')
