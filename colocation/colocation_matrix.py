import pandas as pd

def gen_colocation_matrix(base_results_dir, workload_list, out_file, metric):
    # Generate a colocation matrix for the workloads
    # The matrix has the workloads as rows and columns
    colocation_matrix_columns = ['workload'] + workload_list + ['nothing']
    colocation_matrix = pd.DataFrame(columns=colocation_matrix_columns, dtype=float)
    # Set dtypes
    colocation_matrix = colocation_matrix.astype({'workload': str, 'nothing': float})
    colocation_matrix['workload'] = workload_list
    colocation_matrix.set_index('workload', inplace=True)
    colocation_matrix = colocation_matrix.fillna(0)
    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            if j < i:
                results_dir_str = f'{workload_2}_{workload}'
            else:
                if workload_2 == 'nothing':
                    results_dir_str = workload
                else:
                    results_dir_str = f'{workload}_{workload_2}'
            results_dir = f'{base_results_dir}/{results_dir_str}'
            workload_df = pd.read_csv(f'{results_dir}/workload_data.csv')
            val = workload_df.at[workload_df[workload_df['workload'] == workload].index[0], metric]
            colocation_matrix.at[workload, workload_2] = val
    colocation_matrix.to_csv(out_file, index=True)
    return colocation_matrix

def gen_change_matrix(colocation_matrix_file, out_file):
    colocation_matrix = pd.read_csv(colocation_matrix_file)
    cf_change_matrix = pd.DataFrame(columns=colocation_matrix.columns)
    cf_change_matrix['workload'] = colocation_matrix['workload']
    workload_list = colocation_matrix['workload'].tolist()
    cf_change_matrix.set_index('workload', inplace=True)
    colocation_matrix.set_index('workload', inplace=True)
    
    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            cf_change_matrix.at[workload, workload_2] = colocation_matrix.at[workload, workload_2] - colocation_matrix.at[workload, 'nothing']
    cf_change_matrix.to_csv(out_file, index=True)
    return cf_change_matrix

def gen_relative_change_matrix(colocation_matrix_file, out_file):
    colocation_matrix = pd.read_csv(colocation_matrix_file)
    cf_change_matrix = pd.DataFrame(columns=colocation_matrix.columns)
    cf_change_matrix['workload'] = colocation_matrix['workload']
    workload_list = colocation_matrix['workload'].tolist()
    cf_change_matrix.set_index('workload', inplace=True)
    colocation_matrix.set_index('workload', inplace=True)
    
    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            cf_change_matrix.at[workload, workload_2] = (colocation_matrix.at[workload, workload_2] - colocation_matrix.at[workload, 'nothing']) / colocation_matrix.at[workload, 'nothing']
    cf_change_matrix.to_csv(out_file, index=True)
    return cf_change_matrix

def gen_iso_runtime_normalized_matrix(colocation_matrix_file, runtime_matrix_file, out_file):
    colocation_matrix = pd.read_csv(colocation_matrix_file)
    runtime_matrix = pd.read_csv(runtime_matrix_file)
    iso_runtime_normalized_matrix = pd.DataFrame(columns=colocation_matrix.columns)
    iso_runtime_normalized_matrix['workload'] = colocation_matrix['workload']
    workload_list = colocation_matrix['workload'].tolist()
    iso_runtime_normalized_matrix.set_index('workload', inplace=True)
    colocation_matrix.set_index('workload', inplace=True)
    runtime_matrix.set_index('workload', inplace=True)

    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            iso_runtime_normalized_matrix.at[workload, workload_2] = colocation_matrix.at[workload, workload_2] / runtime_matrix.at[workload, 'nothing']
    iso_runtime_normalized_matrix.to_csv(out_file, index=True)
    return iso_runtime_normalized_matrix