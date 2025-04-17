import os
import pandas as pd
import datetime
import re

def process_docker_logs(log, output):
    csv = {
        'time': [],
        'workload': [],
        'cpu (%)': [],
        'mem_usage': [],
        'mem_limit': [],
        'mem_%': [],
        'net_i/o_used': [],
        'net_i/o_total': [],
        'block_i/o_used': [],
        'block_i/o_total': []
    }
    time = ''
    with open(log) as file:
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
                if len(parts) < 14:
                    continue
                else:
                    if (parts[1].split('_')[0] == 'pbbs'):
                        csv['workload'].append(parts[1].split('_')[1])
                    else:
                        csv['workload'].append(parts[1].split('_')[0])
                    csv['cpu (%)'].append(parts[2].split('%')[0])
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
            # Convert net i/o to GiB
            if 'kB' in row['net_i/o_used']:
                df.at[index, 'net_i/o_used'] = float(row['net_i/o_used'].split('kB')[0]) / 1024 / 1024
            elif 'MB' in row['net_i/o_used']:
                df.at[index, 'net_i/o_used'] = float(row['net_i/o_used'].split('MB')[0]) / 1024
            elif 'GB' in row['net_i/o_used']:
                df.at[index, 'net_i/o_used'] = float(row['net_i/o_used'].split('GB')[0])
            elif 'B' in row['net_i/o_used']:
                df.at[index, 'net_i/o_used'] = float(row['net_i/o_used'].split('B')[0]) / 1024 / 1024 / 1024
            if 'kB' in row['net_i/o_total']:
                df.at[index, 'net_i/o_total'] = float(row['net_i/o_total'].split('kB')[0]) / 1024 / 1024
            elif 'MB' in row['net_i/o_total']:
                df.at[index, 'net_i/o_total'] = float(row['net_i/o_total'].split('MB')[0]) / 1024
            elif 'GB' in row['net_i/o_total']:
                df.at[index, 'net_i/o_total'] = float(row['net_i/o_total'].split('GB')[0])
            elif 'B' in row['net_i/o_total']:
                df.at[index, 'net_i/o_total'] = float(row['net_i/o_total'].split('B')[0]) / 1024 / 1024 / 1024
            # Convert block i/o to GB
            if 'kB' in row['block_i/o_used']:
                df.at[index, 'block_i/o_used'] = float(row['block_i/o_used'].split('kB')[0]) / 1024 / 1024
            elif 'MB' in row['block_i/o_used']:
                df.at[index, 'block_i/o_used'] = float(row['block_i/o_used'].split('MB')[0]) / 1024
            elif 'GB' in row['block_i/o_used']:
                df.at[index, 'block_i/o_used'] = float(row['block_i/o_used'].split('GB')[0])
            elif 'B' in row['block_i/o_used']:
                df.at[index, 'block_i/o_used'] = float(row['block_i/o_used'].split('B')[0]) / 1024 / 1024 / 1024
            if 'kB' in row['block_i/o_total']:
                df.at[index, 'block_i/o_total'] = float(row['block_i/o_total'].split('kB')[0]) / 1024 / 1024
            elif 'MB' in row['block_i/o_total']:
                df.at[index, 'block_i/o_total'] = float(row['block_i/o_total'].split('MB')[0]) / 1024
            elif 'GB' in row['block_i/o_total']:
                df.at[index, 'block_i/o_total'] = float(row['block_i/o_total'].split('GB')[0])
            elif 'B' in row['block_i/o_total']:
                df.at[index, 'block_i/o_total'] = float(row['block_i/o_total'].split('B')[0]) / 1024 / 1024 / 1024

        # Remove rows of df where time does not follow the correct format
        df = df[df['time'].str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}$')]
        df.to_csv(output, index=False)

def per_workload_time_series(docker_log_file, pbbs_file, file_to_write, runtime):
    df_docker = pd.read_csv(docker_log_file)
    df_pbbs = pd.read_csv(pbbs_file)

    df_pbbs = df_pbbs.sort_values(by='start_seconds_offset', ascending=True)
    # Find the earliest starting workload
    earliest_workload = df_pbbs['label'].iloc[0]
    workload_start = df_pbbs['start_seconds_offset'].iloc[0]
    # Find the corresponding row in df
    earliest_row = df_pbbs[df_pbbs['label'] == earliest_workload]
    # Find the start time of the earliest starting workload
    earliest_start = earliest_row['start_seconds']
    # Calculate the time offset
    offset = (workload_start - earliest_start).iloc[0]

    df_pbbs = df_pbbs.sort_values(['end_seconds_offset'], ascending=False)
    workload_end = df_pbbs['end_seconds_offset'].iloc[0]

    df_docker['Datetime'] = pd.to_datetime(df_docker['Datetime'])
    # Reference date for converting to seconds (2000-01-01 00:00:00.0000)
    ref_date = pd.Timestamp('2024-10-01')
    ref_datetime = ref_date.to_datetime64()
    # Convert start and end times to seconds
    df_docker['Time (s) Unadjusted'] = (df_docker['Datetime'] - ref_datetime).dt.total_seconds()
    df_docker['Time (s)'] = df_docker['Time (s) Unadjusted'] + offset
    df_docker = df_docker.drop(['Datetime', 'Time elapsed(s)'], axis=1)

    # Truncate the PCM log to the duration of the permutation
    df_docker = df_docker[df_docker['Time (s)'] >= workload_start]

    # Round time to nearest second
    # pcm_offset = workload_start - df_docker['Time (s)'].iloc[0]

    time = workload_start
    for index, row in df_docker.iterrows():
        df_docker.at[index, 'Time (s)'] = time
        time += 1
        
    df_docker = df_docker[df_docker['Time (s)'] <= workload_end]
    df_docker.to_csv(file_to_write, index=False)