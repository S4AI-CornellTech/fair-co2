import os
import pandas as pd
import datetime
from statistics import geometric_mean

data_dir = './data'
pbbs_workload_list = [
        'removeDuplicates',
        'breadthFirstSearch',
        'minSpanningForest',
        'wordCounts',
        'suffixArray',
        'convexHull',
        'nearestNeighbors',
        'nBody',
    ]

def is_pbbs_workload(workload):
    return workload in pbbs_workload_list

def is_faiss_workload(workload):
    return workload.startswith('faiss')

def is_pgbench_workload(workload):
    return workload.startswith('pgbench')

def is_x265_workload(workload):
    return workload.startswith('x265')

def is_llama_workload(workload):
    return workload.startswith('llama')

def is_spark_workload(workload):
    return workload.startswith('spark')

def find_pbbs_start_end_time(data_dir, perm):
    pbbs_df = pd.read_csv(f'{data_dir}/pbbs.csv')
    start = pbbs_df['start'].min()
    end = pbbs_df['end'].max()
    return start, end

def find_pbbs_runtime(data_dir, workload, workload_start, workload_end, start, end):
    pbbs_df = pd.read_csv(f'{data_dir}/pbbs.csv')
    # find the geomean of the row where workload is the same as the workload
    print(workload)
    pbbs_df = pbbs_df[pbbs_df['label'] == workload]
    runtime = pbbs_df['geomean'].values[0]
    return runtime

def find_faiss_start_end_time(data_dir, perm):
    faiss_df = pd.read_csv(f'{data_dir}/faiss_times_ivf.csv')
    start = faiss_df['query_start'].min()
    end = faiss_df['query_end'].max()
    return start, end

def find_faiss_runtime(data_dir, perm, workload_start, workload_end, start, end):
    faiss_df = pd.read_csv(f'{data_dir}/faiss_times_ivf.csv')
    faiss_df['query_start'] = pd.to_datetime(faiss_df['query_start'])
    faiss_df['query_end'] = pd.to_datetime(faiss_df['query_end'])
    # Filter for rows where the query start time is greater than the start time
    faiss_df = faiss_df[faiss_df['query_start'] >= start]
    # Filter for rows where the query end time is less than the end time
    faiss_df = faiss_df[faiss_df['query_end'] <= end]
    faiss_df['query_time'] = faiss_df['query_end'] - faiss_df['query_start']
    if perm == 'faiss':
        # Only look at the last 100 queries
        faiss_df = faiss_df.tail(100)
    runtime = faiss_df['query_time'].mean().total_seconds()
    return runtime

def find_x265_start_end_time(data_dir, perm):
    x265_df = pd.read_csv(f'{data_dir}/x265.csv') 
    start = x265_df['start_time'].min()
    end = x265_df['end_time'].max()
    return start, end

def find_x265_runtime(data_dir, perm, workload_start, workload_end, start, end):
    x265_df = pd.read_csv(f'{data_dir}/x265.csv')
    x265_df['start_time'] = pd.to_datetime(x265_df['start_time'])
    x265_df['end_time'] = pd.to_datetime(x265_df['end_time'])
    # Filter just first three rounds
    if perm == 'x265':
        print('x265')
        x265_df = x265_df.tail(3)
    else:
        x265_df = x265_df.head(3)
    runtime = x265_df['time'].mean()
    return runtime

def find_llama_start_end_time(data_dir, perm):
    # Read llama.txt as a json file
    llama_df = pd.read_json(f'{data_dir}/llama.txt')
    start = llama_df['test_time'].min()
    start = start.tz_localize(None)
    test_time = sum(llama_df['samples_ns'][0]) / 1e9
    end = start + datetime.timedelta(0, test_time)
    return start, end

def find_llama_runtime(data_dir, perm, workload_start, workload_end, start, end):
    # Read llama.txt as a json file
    llama_df = pd.read_json(f'{data_dir}/llama.txt')
    # times = llama_df['samples_ns'].apply(lambda x: sum(x) / 1e9)
    times = llama_df['samples_ns'][0]
    filtered_times = []
    # Skip the early rounds that are not part of the workload
    time_skipped_at_start = 0
    time_needed_to_skip_at_start = start - workload_start
    time_skipped_at_end = 0
    time_needed_to_skip_at_end = workload_end - end
    skipped_at_start=0
    skipped_at_end=0
    for time in times:
        if time_skipped_at_start >= time_needed_to_skip_at_start.total_seconds():
            break
        time_skipped_at_start += time
        skipped_at_start += 1
    # Skip the late rounds that are not part of the workload
    for time in reversed(times):
        if time_skipped_at_end >= time_needed_to_skip_at_end.total_seconds():
            break
        time_skipped_at_end += time
        skipped_at_end += 1
    filtered_times = times[skipped_at_start:len(times)-skipped_at_end]
    runtime = geometric_mean(filtered_times)
    runtime = runtime / 1e9
    return runtime

def find_spark_start_end_time(data_dir, perm):
    with open(f'{data_dir}/spark.txt') as f:
        i = 1
        for line in f:
            if i == 1:
                start = float(line.split()[-1])
            elif i == 11:
                end = float(line.split()[-1])
            i += 1
    # Convert unix seconds to datetime
    start = datetime.datetime.fromtimestamp(start)
    end = datetime.datetime.fromtimestamp(end)
    # Add 5 hours to the start and end times (hard coded for now)
    start = start + datetime.timedelta(hours=5)
    end = end + datetime.timedelta(hours=5)
    return start, end

def find_spark_runtime(data_dir, perm, workload_start, workload_end, start, end):
    data = {
        'round': [],
        'start': [],
        'end': [],
        'time': []
    }
    with open(f'{data_dir}/spark.txt') as f:
        i = 1
        for line in f:
            if i % 3 == 1:
                start = float(line.split()[-1])
                start = datetime.datetime.fromtimestamp(start)
                start = start + datetime.timedelta(hours=5)
            elif i % 3 == 2:
                end = float(line.split()[-1])
                end = datetime.datetime.fromtimestamp(end)
                end = end + datetime.timedelta(hours=5)
            elif i % 3 == 0:
                time = float(line.split()[-1])
                data['round'].append(i//3)
                data['start'].append(start)
                data['end'].append(end)
                data['time'].append(time)
            i += 1
    df = pd.DataFrame(data)
    # Remove last round
    df = df[:-1]
    runtime = geometric_mean(df['time'])
    return runtime

def find_start_end_time(data_dir, perm):
    csv = {
        'workload': [],
        'cpu' : [],
        'memory': [],
        'start': [],
        'end': []
    }
    # If perm is not a list, convert it to a list
    if not isinstance(perm, list):
        perm = [perm]
    for workload in perm:
        # If the workload is a PBBS workload
        if is_pbbs_workload(workload):
            start, end = find_pbbs_start_end_time(data_dir, perm)
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
        # If the workload is a Faiss workload
        elif is_faiss_workload(workload):
            start, end = find_faiss_start_end_time(data_dir, perm)
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
        # If the workload is a Pgbench workload
        elif is_pgbench_workload(workload):
            start = 0
            end = 0
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
            print('Pgbench workload -- fixed later')
        # If the workload is a X265 workload
        elif is_x265_workload(workload):
            start, end = find_x265_start_end_time(data_dir, perm)
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
        # If the workload is a Llama workload
        elif is_llama_workload(workload):
            start, end = find_llama_start_end_time(data_dir, perm)
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
        # If the workload is a Spark workload
        elif is_spark_workload(workload):
            start, end = find_spark_start_end_time(data_dir, perm)
            csv['workload'].append(workload)
            csv['cpu'].append(48)
            csv['memory'].append(96)
            csv['start'].append(start)
            csv['end'].append(end)
        else:
            print(f'Unknown workload: {workload}')
    
    df = pd.DataFrame(csv)
    df.to_csv(f'{data_dir}/workload_times.csv')
    return

def fix_pgbench_start_end_time(data_dir, perm):
    pgbench_runtime = datetime.timedelta(seconds=250)
    df_times = pd.read_csv(f'{data_dir}/workload_times.csv')
    flag = 0
    for index, row in df_times.iterrows():
        workload = row['workload']
        if (is_pgbench_workload(workload) != True):
            start = row['start']
            start = pd.to_datetime(start)
            flag = 1
    for index, row in df_times.iterrows():
        workload = row['workload']
        if is_pgbench_workload(workload) and flag == 1:
            df_times.at[index, 'start'] = start
            df_times.at[index, 'end'] = start + pgbench_runtime
    # If all workloads in perm are pgbench, then read start file
    if flag == 0:
        print('All workloads are pgbench')
        # read start csv
        start_df = pd.read_csv(f'{data_dir}/start')
        start = start_df['time'].min()
        # Start is datetime format
        start = pd.to_datetime(start)
        for index, row in df_times.iterrows():
            df_times.at[index, 'start'] = start
            df_times.at[index, 'end'] = start + pgbench_runtime
    df_times.to_csv(f'{data_dir}/workload_times.csv')
    return

def find_pgbench_runtime(data_dir, workload):
    total_runtime = 250
    # Open txt file name starting with pgbench
    if is_pgbench_workload(workload):
        with open(f'{data_dir}/{workload}.txt') as f:
            for line in f:
                if 'number of transactions actually processed:' in line:
                    number_transactions = float(line.split()[-1])
            runtime = total_runtime / (number_transactions / 1000000) # Print runtime per 1000000 transactions
    return runtime                

def find_avg_cpu_util(data_dir, perm):
    df_times = pd.read_csv(f'{data_dir}/workload_times.csv')
    docker_file = f'{data_dir}/docker_processed.csv'
    df_docker = pd.read_csv(docker_file)
    df_docker['time'] = pd.to_datetime(df_docker['time'])
    start = df_times['start'].max()
    end = df_times['end'].min()
    df_docker = df_docker[(df_docker['time'] >= start) & (df_docker['time'] <= end)]
    #print(df_docker)
    for index, row in df_times.iterrows():
        workload = row['workload']
        #print(workload)
        df_workload = df_docker[df_docker['workload'] == workload]
        average_cpu = df_workload['cpu (%)'].mean()
        df_times.at[index, 'cpu_util_avg'] = average_cpu / 100
    df_times.to_csv(f'{data_dir}/workload_data.csv')
    return

def find_avg_runtime(data_dir, perm):
    df_data = pd.read_csv(f'{data_dir}/workload_data.csv')
    df_data['start'] = pd.to_datetime(df_data['start'], format='mixed')
    df_data['end'] = pd.to_datetime(df_data['end'], format='mixed')
    start = df_data['start'].max()
    end = df_data['end'].min()
    for index, row in df_data.iterrows():
        workload = row['workload']
        workload_start = row['start']
        workload_start = pd.to_datetime(workload_start)
        workload_end = row['end']
        workload_end = pd.to_datetime(workload_end)
        if is_pbbs_workload(workload):
            runtime = find_pbbs_runtime(data_dir, workload, workload_start, workload_end, start, end)
        elif is_faiss_workload(workload):
            runtime = find_faiss_runtime(data_dir, perm, workload_start, workload_end, start, end)
        elif is_pgbench_workload(workload):
            runtime = find_pgbench_runtime(data_dir, workload)
        elif is_x265_workload(workload):
            runtime = find_x265_runtime(data_dir, perm, workload_start, workload_end, start, end)
        elif is_llama_workload(workload):
            runtime = find_llama_runtime(data_dir, perm, workload_start, workload_end, start, end)
        elif is_spark_workload(workload):
            runtime = find_spark_runtime(data_dir, perm, workload_start, workload_end, start, end)
        df_data.at[index, 'runtime'] = runtime
    df_data.to_csv(f'{data_dir}/workload_data.csv')
    return

def find_avg_power_energy(data_dir, static_cpu_power, static_dram_power):
    df_data = pd.read_csv(f'{data_dir}/workload_data.csv')
    pcm_df = pd.read_csv(f'{data_dir}/pcm_processed.csv')
    pcm_df['Datetime'] = pd.to_datetime(pcm_df['Datetime'])
    pcm_df = pcm_df[(pcm_df['Datetime'] >= df_data['start'].max()) & (pcm_df['Datetime'] <= df_data['end'].min())]
    df_data['total CPU power (W)'] = pcm_df['Proc Energy (Joules)'].sum() / (pcm_df['Datetime'].max()- pcm_df['Datetime'].min()).total_seconds()
    df_data['total DRAM power (W)'] = pcm_df['DRAM Energy (Joules)'].sum() / (pcm_df['Datetime'].max()- pcm_df['Datetime'].min()).total_seconds()
    df_data['total power (W)'] = df_data['total CPU power (W)'] + df_data['total DRAM power (W)']
    df_data['total CPU dynamic power (W)'] = df_data['total CPU power (W)'] - static_cpu_power
    df_data['total DRAM dynamic power (W)'] = df_data['total DRAM power (W)'] - static_dram_power
    df_data['total dynamic power (W)'] = df_data['total CPU dynamic power (W)'] + df_data['total DRAM dynamic power (W)']
    df_data['proportional CPU dynamic power (W)'] = df_data['total CPU dynamic power (W)'] * df_data['cpu_util_avg'] / df_data['cpu_util_avg'].sum()
    df_data['proportional DRAM dynamic power (W)'] = df_data['total DRAM dynamic power (W)'] * df_data['cpu_util_avg'] / df_data['cpu_util_avg'].sum()
    df_data['proportional dynamic power (W)'] = df_data['total dynamic power (W)'] * df_data['cpu_util_avg'] / df_data['cpu_util_avg'].sum()
    df_data['proportional CPU static power (W)'] = static_cpu_power * df_data['cpu'] / df_data['cpu'].sum()
    df_data['proportional DRAM static power (W)'] = static_dram_power * df_data['memory'] / df_data['memory'].sum()
    df_data['proportional static power (W)'] = df_data['proportional CPU static power (W)'] + df_data['proportional DRAM static power (W)']
    df_data['proportional CPU static energy (J)'] = df_data['proportional CPU static power (W)'] * df_data['runtime']
    df_data['proportional DRAM static energy (J)'] = df_data['proportional DRAM static power (W)'] * df_data['runtime']
    df_data['proportional static energy (J)'] = df_data['proportional CPU static energy (J)'] + df_data['proportional DRAM static energy (J)']
    df_data['proportional CPU dynamic energy (J)'] = df_data['proportional CPU dynamic power (W)'] * df_data['runtime']
    df_data['proportional DRAM dynamic energy (J)'] = df_data['proportional DRAM dynamic power (W)'] * df_data['runtime']
    df_data['proportional dynamic energy (J)'] = df_data['proportional CPU dynamic energy (J)'] + df_data['proportional DRAM dynamic energy (J)']
    df_data['proportional total energy (J)'] = df_data['proportional static energy (J)'] + df_data['proportional dynamic energy (J)']
    # Drop Unamed columns
    df_data = df_data.loc[:, ~df_data.columns.str.contains('^Unnamed')]
    df_data.to_csv(f'{data_dir}/workload_data.csv')
    return

def process_logs(data_dir, perm, static_cpu_power, static_dram_power):
    find_start_end_time(data_dir, perm)
    fix_pgbench_start_end_time(data_dir, perm)
    find_avg_cpu_util(data_dir, perm)
    find_avg_runtime(data_dir, perm)
    find_avg_power_energy(data_dir, static_cpu_power, static_dram_power)
    return

def process_pbbs_logs(data_dir):
    csv = {
        'workload': [],
        'label': [],
        'rounds': [],
        'cpus': [],
        'memory': [],
        'times': [],
        'geomean': [],
        'start': [],
        'end': []
    }
    # PBBS logs are in the format pbbs_{workload}_{cpus}_{memory}.log
    pbbs_logs = [log for log in os.listdir(data_dir) if log.startswith('pbbs') and log.endswith('.log')]
    for log in pbbs_logs:
        params = log.split('_')    
        workload_name = params[1]
        if workload_name == 'breadthFirstSearch':
            workload = 'backForwardBFS'
        elif workload_name == 'minSpanningForest':
            workload = 'parallelFilterKruskal'
        elif workload_name == 'convexHull':
            workload = 'quickHull'
        elif workload_name == 'nBody':
            workload = 'parallelCK'
        elif workload_name == 'nearestNeighbors':
            workload = 'octTree'
        elif workload_name == 'wordCounts':
            workload = 'histogram'
        elif workload_name == 'removeDuplicates':
            workload = 'parlayhash'
        elif workload_name == 'suffixArray':
            workload = 'parallelKS'
        cpus = int(params[-2])
        memory = int(params[-1].split('.')[0])
        with open(f'{data_dir}/{log}') as file:
            # skip the first 12 lines
            for _ in range(6):
                line = next(file)
            line = next(file)
            # print(line)
            # first word is the input if it is not equal to workload name
            input = line.split()[0]
            if workload == 'octTree':
                k = line.split('-k')[1].split()[0]
                input = f'{input}_k{k}'
            # add the input to the list
            csv['workload'].append(workload)
            csv['label'].append(workload_name)
            # read the number of rounds as denoted by -r
            rounds = line.split('-r')[1].split()[0]
            rounds = int(rounds)
            # add the number of rounds to the list
            csv['rounds'].append(rounds)
            # add the cpus to the list
            csv['cpus'].append(cpus)
            # add the memory to the list
            csv['memory'].append(memory)
            times_str = line.split(' : ')[2].split(', ')[:rounds//2 + 1]
            times = [float(time.split("'")[1]) for time in times_str]
            csv['times'].append(times)
            # read the geomean value
            geomean = geometric_mean(times)
            # add the geomean to the list
            csv['geomean'].append(geomean)
            # read the start time
            start = line.split('start = ')[1].split(',')[0]
            start = pd.to_datetime(start)
            # add the start time to the list
            csv['start'].append(start)
            # read the end time
            end = start + datetime.timedelta(0,sum(times))
            # add the end time to the list
            csv['end'].append(end)

    df = pd.DataFrame(csv)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    # Reference date for converting to seconds (2000-01-01 00:00:00.0000)
    ref_datetime = df['start'].min()
    # Convert start and end times to seconds
    df['start_seconds'] = (df['start'] - ref_datetime).dt.total_seconds()
    df['end_seconds'] = (df['end'] - ref_datetime).dt.total_seconds()
    # Calculate duration of each workload
    df['duration'] = (df['end'] - df['start']).dt.total_seconds()
    df.to_csv(f'{data_dir}/pbbs.csv')
    return df

def find_embodied_cf(data_dir, cpu_ci, mem_ci):
    df = pd.read_csv(f'{data_dir}/workload_data.csv')
    cpu_cf = cpu_ci * 96
    mem_cf = mem_ci * 192
    df['CPU Embodied Carbon (gCO2e)'] = df['cpu'] / df['cpu'].sum() * cpu_cf
    df['Memory Embodied Carbon (gCO2e)'] = df['memory'] / df['memory'].sum() * mem_cf
    df['Total Embodied Carbon (gCO2e)'] = df['CPU Embodied Carbon (gCO2e)'] + df['Memory Embodied Carbon (gCO2e)']
    # Drop Unamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv(f'{data_dir}/workload_data.csv')
    return

def get_per_workload_stats(data_dir, static_cpu_power, static_dram_power):
    pbbs_file = f'{data_dir}/pbbs.csv'
    df = pd.read_csv(pbbs_file)
    # Find CPU-seconds, GB-seconds, utilization-seconds (% seconds)
    df['cpu_seconds'] = df['cpus'] * df['geomean']
    df['memory_seconds'] = df['memory'] * df['geomean']
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    docker_file = f'{data_dir}/docker_processed.csv'
    df_docker = pd.read_csv(docker_file)
    df_docker['time'] = pd.to_datetime(df_docker['time'])
    #print(df_docker)
    for index, row in df.iterrows():
        workload = row['label']
        #print(workload)
        df_workload = df_docker[df_docker['workload'] == workload]
        df_workload = df_workload[(df_workload['time'] >= row['start']) & (df_workload['time'] <= row['end'])]
        average_cpu = df_workload['cpu (%)'].mean()
        geomean = row['geomean']
        #print(average_cpu)
        df.at[index, 'cpu_util_avg'] = average_cpu / 100
        df.at[index, 'cpu_util_seconds'] = average_cpu / 100 * geomean

    pcm_df = pd.read_csv(f'{data_dir}/pcm_truncated.csv')
    df['total CPU power (W)'] = pcm_df['Proc Energy (Joules)'].sum() / (pcm_df['Time (s)'].max()- pcm_df['Time (s)'].min())
    df['total DRAM power (W)'] = pcm_df['DRAM Energy (Joules)'].sum() / (pcm_df['Time (s)'].max()- pcm_df['Time (s)'].min())
    df['total power (W)'] = df['total CPU power (W)'] + df['total DRAM power (W)']
    df['total CPU dynamic power (W)'] = df['total CPU power (W)'] - static_cpu_power
    df['total DRAM dynamic power (W)'] = df['total DRAM power (W)'] - static_dram_power
    df['total dynamic power (W)'] = df['total CPU dynamic power (W)'] + df['total DRAM dynamic power (W)']
    df['proportional CPU dynamic power (W)'] = df['total CPU dynamic power (W)'] * df['cpu_util_avg'] / df['cpu_util_avg'].sum()
    df['proportional DRAM dynamic power (W)'] = df['total DRAM dynamic power (W)'] * df['cpu_util_avg'] / df['cpu_util_avg'].sum()
    df['proportional dynamic power (W)'] = df['total dynamic power (W)'] * df['cpu_util_avg'] / df['cpu_util_avg'].sum()
    df['proportional CPU static power (W)'] = static_cpu_power * df['cpus'] / df['cpus'].sum()
    df['proportional DRAM static power (W)'] = static_dram_power * df['memory'] / df['memory'].sum()
    df['proportional static power (W)'] = df['proportional CPU static power (W)'] + df['proportional DRAM static power (W)']
    df['proportional CPU static energy (J)'] = df['proportional CPU static power (W)'] * df['geomean']
    df['proportional DRAM static energy (J)'] = df['proportional DRAM static power (W)'] * df['geomean']
    df['proportional static energy (J)'] = df['proportional CPU static energy (J)'] + df['proportional DRAM static energy (J)']
    df['proportional CPU dynamic energy (J)'] = df['proportional CPU dynamic power (W)'] * df['geomean']
    df['proportional DRAM dynamic energy (J)'] = df['proportional DRAM dynamic power (W)'] * df['geomean']
    df['proportional dynamic energy (J)'] = df['proportional CPU dynamic energy (J)'] + df['proportional DRAM dynamic energy (J)']
    df['proportional total energy (J)'] = df['proportional static energy (J)'] + df['proportional dynamic energy (J)']

    # Save
    df.to_csv(f'{data_dir}/pbbs_processed.csv')

def find_idle_power(data_dir, interval):
    pcm_logs = os.listdir(f'{data_dir}/pcm_processed')
    cpu_idle_power = 0
    dram_idle_power = 0
    time = 0
    for pcm_log in pcm_logs:
        # Load the PCM log in a dataframe
        df_pcm = pd.read_csv(f'{data_dir}/pcm_processed/{pcm_log}')
        # Filter for rows where the CPU utilization is less sthan 20 percent
        df_pcm = df_pcm[df_pcm['CPU Utilization (%)'] < 50]
        # Find the idle power consumed by the CPU
        cpu_idle_power += df_pcm['Proc Energy (Joules)'].sum()
        # Find the idle power consumed by the DRAM
        dram_idle_power += df_pcm['DRAM Energy (Joules)'].sum()
        time += interval * df_pcm.shape[0]
    # Find the average idle power consumed by the CPU
    cpu_idle_power /= time
    # Find the average idle power consumed by the DRAM
    dram_idle_power /= time
    return cpu_idle_power, dram_idle_power