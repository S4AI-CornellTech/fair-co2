import pandas as pd
import os

def process_pcm(log_file, workload, file_to_write, node):
    
    if node == 'storage':
        df2 = pd.read_csv(log_file, skiprows=[0], iterator=True, chunksize=1000) #Dataframe starting at second row (necessary since df1 has no time column)
    elif node == 'clr':
        df2 = pd.read_csv(log_file, skiprows=[0,1,2], iterator=True, chunksize=1000) #Dataframe starting at second row (necessary since df1 has no time column)
    else:
        print('Invalid node type')
        return
    
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
        if node == 'storage':
            num_cores = 40
        elif node == 'clr':
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
            

def truncate_pcm(pcm_log_file, pbbs_file, file_to_write):
    df_pbbs = pd.read_csv(pbbs_file)
    df_pcm = pd.read_csv(pcm_log_file)

    df_pbbs = df_pbbs.sort_values(by='start_seconds', ascending=True)
    # Find the earliest starting workload
    earliest_workload = df_pbbs['label'].iloc[0]
    workload_start = df_pbbs['start_seconds'].iloc[0]
    workload_start_datetime = df_pbbs['start'].iloc[0]
    workload_start_datetime = pd.to_datetime(workload_start_datetime)

    df_pbbs = df_pbbs.sort_values(['end_seconds'], ascending=False)
    workload_end = df_pbbs['end_seconds'].iloc[0]

    df_pcm['Datetime'] = pd.to_datetime(df_pcm['Datetime'])
    # Reference date for converting to seconds (2000-01-01 00:00:00.0000)
    # Convert start and end times to seconds
    df_pcm['Time (s)'] = (df_pcm['Datetime'] - workload_start_datetime).dt.total_seconds()

    # Truncate the PCM log to the duration of the permutation
    df_pcm = df_pcm[df_pcm['Time (s)'] >= workload_start]
        
    df_pcm = df_pcm[df_pcm['Time (s)'] <= workload_end]

    # Drop Time elapsed(s) column
    df_pcm = df_pcm.drop(['Time elapsed(s)'], axis=1)
    df_pcm.to_csv(file_to_write, index=False)

def get_avg_stats(base_results_dir, workload_list):
    avg_stats = {'workload': [], 'Memory Read Bandwidth (System) (GB/s)': [], 'Memory Write Bandwidth (System) (GB/s)': []}
    avg_stats_df = pd.DataFrame(avg_stats)
    avg_stats_df['workload'] = workload_list
    avg_stats_df.set_index('workload', inplace=True)
    for workload in workload_list:
        results_dir = f'{base_results_dir}/{workload}'
        df = pd.read_csv(f'{results_dir}/pcm_truncated.csv')
        avg_stats_df.at[workload, 'Memory Read Bandwidth (System) (GB/s)'] = df['Memory Read Bandwidth (System) (GB/s)'].mean()
        avg_stats_df.at[workload, 'Memory Write Bandwidth (System) (GB/s)'] = df['Memory Write Bandwidth (System) (GB/s)'].mean()
        avg_stats_df.at[workload, 'Memory Bandwidth (System) (GB/s)'] = df['Memory Read Bandwidth (System) (GB/s)'].mean() + df['Memory Write Bandwidth (System) (GB/s)'].mean()
        avg_stats_df.at[workload, 'Memory Utilization (System) (%)'] = df['Memory Utilization (System) (%)'].mean()
        avg_stats_df.at[workload, 'CPU Utilization (%)'] = df['CPU Utilization (%)'].mean()
        avg_stats_df.at[workload, 'Socket 0 L3 Cache Occupancy'] = df['Socket 0 L3 Cache Occupancy'].mean()
        avg_stats_df.at[workload, 'Socket 1 L3 Cache Occupancy'] = df['Socket 1 L3 Cache Occupancy'].mean()
        avg_stats_df.at[workload, 'L3 Cache Occupancy'] = df['Socket 0 L3 Cache Occupancy'].mean() + df['Socket 1 L3 Cache Occupancy'].mean()
        avg_stats_df.at[workload, 'CPU Power (W)'] = df['Proc Energy (Joules)'].sum() / df['Time (s)'].max()
        avg_stats_df.at[workload, 'DRAM Power (W)'] = df['DRAM Energy (Joules)'].sum() / df['Time (s)'].max()
    avg_stats_df.to_csv(f'{base_results_dir}/avg_stats.csv', index=True)
    print(avg_stats_df)