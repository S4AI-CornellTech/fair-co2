import os
import pandas as pd
import random
import concurrent.futures
import warnings
import argparse
warnings.filterwarnings("ignore")

def gen_cf_colocation_matrix(data_dir, workload_list, grid_ci, cpu_ci, mem_ci):
    # Convert grid CI to J
    kWh_to_J = 3.6e6
    grid_ci = grid_ci / kWh_to_J
    # Generate a colocation matrix for the workloads
    # The matrix has the workloads as rows and columns
    colocation_matrix_columns = ['workload'] + workload_list + ['nothing']
    colocation_matrix = pd.DataFrame(columns=colocation_matrix_columns, dtype=float)
    # Set dtypes
    colocation_matrix = colocation_matrix.astype({'workload': str, 'nothing': float})
    colocation_matrix['workload'] = workload_list
    colocation_matrix.set_index('workload', inplace=True)
    colocation_matrix = colocation_matrix.fillna(0)
    # Read in runtime matrix
    runtime_matrix_file = f'{data_dir}/runtime_colocation_matrix.csv'
    runtime_matrix = pd.read_csv(runtime_matrix_file)
    runtime_matrix.set_index('workload', inplace=True)
    # Read in proportional energy matrix
    proportional_energy_matrix_file = f'{data_dir}/proportional_energy_colocation_matrix.csv'
    proportional_energy_matrix = pd.read_csv(proportional_energy_matrix_file)
    proportional_energy_matrix.set_index('workload', inplace=True)
    # Calculate the carbon footprint matrix
    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            runtime = runtime_matrix.at[workload, workload_2]
            energy = proportional_energy_matrix.at[workload, workload_2]
            if workload_2 == 'nothing':
                colocation_matrix.at[workload, workload_2] = (cpu_ci * 96 + mem_ci * 192) * runtime + grid_ci * energy
            else:
                colocation_matrix.at[workload, workload_2] = (cpu_ci * 48 + mem_ci * 96) * runtime + grid_ci * energy
    return colocation_matrix

def gen_iso_runtime_normalized_matrix(colocation_matrix_in, runtime_matrix_in):
    colocation_matrix = colocation_matrix_in.copy()
    runtime_matrix = runtime_matrix_in.copy()
    iso_runtime_normalized_matrix = pd.DataFrame(index=colocation_matrix.index, columns=colocation_matrix.columns)
    workload_list = colocation_matrix.index.tolist()
    runtime_matrix.set_index('workload', inplace=True)

    for i, workload in enumerate(workload_list):
        for j, workload_2 in enumerate(workload_list + ['nothing']):
            iso_runtime_normalized_matrix.at[workload, workload_2] = colocation_matrix.at[workload, workload_2] / runtime_matrix.at[workload, 'nothing']
    return iso_runtime_normalized_matrix


def find_colocation_cf(matrix_df, colocation):
    cf = 0
    for i in range(0, len(colocation), 2):
        workload = colocation[i].split('_')[0]
        if i+1 >= len(colocation):
            workload_2 = 'nothing'
        else:
            workload_2 = colocation[i+1].split('_')[0]
        cf += matrix_df.at[workload, workload_2]
        if workload_2 != 'nothing':
            cf += matrix_df.at[workload_2, workload]
    return cf

def gen_colocation(candidate_workloads, n_workloads):
    colocation = []
    for i in range(0, n_workloads):
        workload = random.sample(candidate_workloads, 1)
        colocation.append(str(workload[0]) + f'_{i}')
    return colocation

def filter_matrix(matrix_df, num_samples, max_samples=15):
    matrix_df_filtered_inflicted = matrix_df.copy()
    matrix_df_filtered_suffered = matrix_df.copy()
    random_columns = random.sample(list(matrix_df.columns), min(num_samples, max_samples))
    random_rows = random.sample(list(matrix_df.index), min(num_samples, max_samples))
    matrix_df_filtered_inflicted = matrix_df_filtered_inflicted.loc[random_rows]
    matrix_df_filtered_suffered = matrix_df_filtered_suffered[random_columns]
    return matrix_df_filtered_inflicted, matrix_df_filtered_suffered

def ground_truth_shapley(data_dir, iso_runtime_normalized_matrix, workload_list_in):
    # Generate a colocation matrix for the workloads
    # The matrix has the workloads as rows and columns
    workload_list = workload_list_in.copy()
    runtime_matrix_file = f'{data_dir}/runtime_colocation_matrix.csv'
    proportional_energy_colocation_matrix_file = f'{data_dir}/proportional_energy_colocation_matrix.csv'
    matrix_df = iso_runtime_normalized_matrix.copy()
    runtime_matrix_df = pd.read_csv(runtime_matrix_file)
    runtime_matrix_df.set_index('workload', inplace=True)
    proportional_energy_colocation_matrix_df = pd.read_csv(proportional_energy_colocation_matrix_file)
    proportional_energy_colocation_matrix_df.set_index('workload', inplace=True)
    shapley_values_df = pd.DataFrame(columns=['workload', 'shapley_attribution', 'baseline_attribution'])
    shapley_values_df['workload'] = workload_list
    shapley_values_df.set_index('workload', inplace=True)
    shapley_values_df['shapley_attribution'] = 0.0
    shapley_values_df['baseline_attribution'] = 0.0
    # Baseline attribution
    for i in range(0, len(workload_list), 2):
        workload_label = workload_list[i]
        workload = workload_list[i].split('_')[0]
        if i+1 >= len(workload_list):
            workload_2_label = 'nothing'
            workload_2 = 'nothing'
        else:
            workload_2_label = workload_list[i+1]
            workload_2 = workload_list[i+1].split('_')[0]
        if workload_2 != 'nothing':
            shapley_values_df.at[workload_label, 'baseline_attribution'] = matrix_df.at[workload, workload_2]
            shapley_values_df.at[workload_2_label, 'baseline_attribution'] = matrix_df.at[workload_2, workload]
            shapley_values_df.at[workload_2_label, 'partner_workload'] = workload_label
            shapley_values_df.at[workload_label, 'Runtime (normalized)'] = runtime_matrix_df.at[workload, workload_2] / runtime_matrix_df.at[workload, 'nothing']
            shapley_values_df.at[workload_2_label, 'Runtime (normalized)'] = runtime_matrix_df.at[workload_2, workload] / runtime_matrix_df.at[workload_2, 'nothing']
        else:
            shapley_values_df.at[workload_label, 'baseline_attribution'] = matrix_df.at[workload, workload_2]
            shapley_values_df.at[workload_label, 'Runtime (normalized)'] = 1
        shapley_values_df.at[workload_label, 'partner_workload'] = workload_2_label

    # Shapley attributions
    num_workloads = len(workload_list)
    if len(workload_list) % 2 == 1:
        workload_list.append('nothing')
    for i in range(0,num_workloads,1):
        workload_label = workload_list[i]
        workload = workload_list[i].split('_')[0]
        isolated_cf = matrix_df.at[workload, 'nothing']
        for j in range(0,len(workload_list),1):
            if i != j:
                partner_workload = workload_list[j].split('_')[0]
                shapley_values_df.at[workload_label, 'shapley_attribution'] += (matrix_df.at[workload, partner_workload])# Own CF when added to partner
                if partner_workload != 'nothing':
                    partner_isolated_cf = matrix_df.at[partner_workload, 'nothing']
                    shapley_values_df.at[workload_label, 'shapley_attribution'] += (matrix_df.at[partner_workload, workload] - partner_isolated_cf) # Partner change in CF 
                shapley_values_df.at[workload_label, 'shapley_attribution'] += isolated_cf # Own CF when added to nothing
    reference_cf = find_colocation_cf(matrix_df, workload_list)
    shapley_values_df['shapley_attribution'] = shapley_values_df['shapley_attribution'] * reference_cf / shapley_values_df['shapley_attribution'].sum()
    
    return shapley_values_df

def interference_adjustment(data_dir, shapley_df, grid_ci, num_samples, verbose = False):
    kWh_to_J = 3.6e6
    grid_ci_J = grid_ci / kWh_to_J
    runtime_relative_change_matrix_file = f'{data_dir}/runtime_relative_change_matrix.csv'
    proportional_energy_relative_change_matrix_file = f'{data_dir}/proportional_dynamic_energy_relative_change_matrix.csv'
    
    workloads_labels = shapley_df.index.tolist() 

    # Rename columns
    shapley_df.rename(columns={'shapley_attribution': 'Shapley Attribution (gCO2eq/iso-s)'}, inplace=True)
    shapley_df.rename(columns={'baseline_attribution': 'Baseline Attribution (gCO2eq/iso-s)'}, inplace=True)   

    runtime_relative_change_matrix_df = pd.read_csv(runtime_relative_change_matrix_file)
    
    proportional_energy_relative_change_matrix_df = pd.read_csv(proportional_energy_relative_change_matrix_file)

    # Find average inflicted and suffered runtime and proportional energy changes
    runtime_relative_change_inflicted_df = pd.DataFrame(columns=['workload', 'avg_suffered_change + 1', 'avg_inflicted_change + 1', 'avg_of_avg'])
    runtime_relative_change_inflicted_df['workload'] = runtime_relative_change_matrix_df['workload']
    runtime_relative_change_matrix_df.set_index('workload', inplace=True)
    runtime_relative_change_inflicted_df.set_index('workload', inplace=True)

    proportional_energy_relative_change_inflicted_df = pd.DataFrame(columns=['workload', 'avg_suffered_change + 1', 'avg_inflicted_change + 1', 'avg_of_avg'])
    proportional_energy_relative_change_inflicted_df['workload'] = proportional_energy_relative_change_matrix_df['workload']
    proportional_energy_relative_change_matrix_df.set_index('workload', inplace=True)
    proportional_energy_relative_change_inflicted_df.set_index('workload', inplace=True)

    # Drop out keep random columns and rows
    runtime_relative_change_matrix_df_filtered_inflicted, runtime_relative_change_matrix_df_filtered_suffered = filter_matrix(runtime_relative_change_matrix_df, num_samples, max_samples=15)
    proportional_energy_relative_change_matrix_df_filtered_inflicted, proportional_energy_relative_change_matrix_df_filtered_suffered = filter_matrix(proportional_energy_relative_change_matrix_df, num_samples, max_samples=15)

    # print(runtime_relative_change_matrix_df_filtered_inflicted)

    avg_runtime_relative_change_inflicted = runtime_relative_change_matrix_df_filtered_inflicted.mean(axis=0)
    avg_runtime_relative_change_suffered = runtime_relative_change_matrix_df_filtered_suffered.mean(axis=1)

    # print(avg_runtime_relative_change_inflicted)

    avg_proportional_energy_relative_change_inflicted = proportional_energy_relative_change_matrix_df_filtered_inflicted.mean(axis=0)
    avg_proportional_energy_relative_change_suffered = proportional_energy_relative_change_matrix_df_filtered_suffered.mean(axis=1)

    runtime_relative_change_inflicted_df['avg_suffered_change + 1'] = avg_runtime_relative_change_suffered + 1
    runtime_relative_change_inflicted_df['avg_inflicted_change + 1'] = avg_runtime_relative_change_inflicted + 1
    proportional_energy_relative_change_inflicted_df['avg_suffered_change + 1'] = avg_proportional_energy_relative_change_suffered + 1
    proportional_energy_relative_change_inflicted_df['avg_inflicted_change + 1'] = avg_proportional_energy_relative_change_inflicted + 1

    # Find average of averages
    runtime_relative_change_inflicted_df['avg_of_avg'] = (runtime_relative_change_inflicted_df['avg_suffered_change + 1'] + runtime_relative_change_inflicted_df['avg_inflicted_change + 1']) / 2
    proportional_energy_relative_change_inflicted_df['avg_of_avg'] = (proportional_energy_relative_change_inflicted_df['avg_suffered_change + 1'] + proportional_energy_relative_change_inflicted_df['avg_inflicted_change + 1']) / 2

    # Find iso runtime normalized embodied carbon sum, static energy sum, and dynamic energy sum
    iso_runtime_normalized_embodied_cf_matrix_file = f'{data_dir}/iso_runtime_normalized_embodied_cf_matrix.csv'
    iso_runtime_normalized_static_energy_matrix_file = f'{data_dir}/iso_runtime_normalized_static_energy_matrix.csv'
    iso_runtime_normalized_energy_matrix_file = f'{data_dir}/iso_runtime_normalized_energy_matrix.csv'
    iso_runtime_normalized_embodied_cf_matrix_df = pd.read_csv(iso_runtime_normalized_embodied_cf_matrix_file)
    iso_runtime_normalized_embodied_cf_matrix_df.set_index('workload', inplace=True)
    iso_runtime_normalized_static_energy_matrix_df = pd.read_csv(iso_runtime_normalized_static_energy_matrix_file)
    iso_runtime_normalized_static_energy_matrix_df.set_index('workload', inplace=True)
    iso_runtime_normalized_energy_matrix_df = pd.read_csv(iso_runtime_normalized_energy_matrix_file)
    iso_runtime_normalized_energy_matrix_df.set_index('workload', inplace=True)

    for workload_label in workloads_labels:
        partner_workload_label = shapley_df.at[workload_label, 'partner_workload']
        workload = workload_label.split('_')[0]
        partner_workload = partner_workload_label.split('_')[0]
        shapley_df.at[workload_label, 'Average of Average Suffered and Inflicted Normalized Runtime'] = runtime_relative_change_inflicted_df.at[workload, 'avg_of_avg']
        shapley_df.at[workload_label, 'Average of Average Suffered and Inflicted Normalized Proportional Energy'] = proportional_energy_relative_change_inflicted_df.at[workload, 'avg_of_avg']
        shapley_df.at[workload_label, 'Embodied CF (gCO2eq/iso-s)'] = iso_runtime_normalized_embodied_cf_matrix_df.at[workload, partner_workload]
        shapley_df.at[workload_label, 'Static Energy (J/iso-s)'] = iso_runtime_normalized_static_energy_matrix_df.at[workload, partner_workload]
        shapley_df.at[workload_label, 'Energy (J/iso-s)'] = iso_runtime_normalized_energy_matrix_df.at[workload, partner_workload]
        shapley_df.at[workload_label, 'Isolated Energy (J/iso-s)'] = iso_runtime_normalized_energy_matrix_df.at[workload, 'nothing']
        shapley_df.at[workload_label, 'Isolated Static Energy (J/iso-s)'] = iso_runtime_normalized_static_energy_matrix_df.at[workload, 'nothing']
        shapley_df.at[workload_label, 'Isolated Dynamic Energy (J/iso-s)'] = iso_runtime_normalized_energy_matrix_df.at[workload, 'nothing'] - iso_runtime_normalized_static_energy_matrix_df.at[workload, 'nothing']
    
    embodied_cf_sum = shapley_df['Embodied CF (gCO2eq/iso-s)'].sum()
    static_energy_sum = shapley_df['Static Energy (J/iso-s)'].sum()
    energy_sum = shapley_df['Energy (J/iso-s)'].sum()
    dynamic_energy_sum = energy_sum - static_energy_sum
    average_of_average_suffered_and_inflicted_normalized_runtime_sum = shapley_df['Average of Average Suffered and Inflicted Normalized Runtime'].sum()
    
    shapley_df['Energy Attribution Factor'] = shapley_df['Isolated Dynamic Energy (J/iso-s)'] * shapley_df['Average of Average Suffered and Inflicted Normalized Proportional Energy']
    shapley_df['Embodied CF Adjusted (gCO2eq/iso-s)'] = shapley_df['Average of Average Suffered and Inflicted Normalized Runtime'] / average_of_average_suffered_and_inflicted_normalized_runtime_sum * embodied_cf_sum
    shapley_df['Static Energy Adjusted (gCO2eq/iso-s)'] = shapley_df['Average of Average Suffered and Inflicted Normalized Runtime'] / average_of_average_suffered_and_inflicted_normalized_runtime_sum * static_energy_sum
    energy_attribution_factor_sum = shapley_df['Energy Attribution Factor'].sum()
    shapley_df['Dynamic Energy Adjusted (J/iso-s)'] = shapley_df['Energy Attribution Factor'] / energy_attribution_factor_sum * dynamic_energy_sum

    shapley_df['Attribution Adjusted (gCO2eq/iso-s)'] = shapley_df['Embodied CF Adjusted (gCO2eq/iso-s)'] + shapley_df['Static Energy Adjusted (gCO2eq/iso-s)'] * grid_ci_J + shapley_df['Dynamic Energy Adjusted (J/iso-s)'] * grid_ci_J

    shapley_df = shapley_df[['partner_workload', 'Runtime (normalized)', 'Baseline Attribution (gCO2eq/iso-s)', 'Shapley Attribution (gCO2eq/iso-s)', 'Attribution Adjusted (gCO2eq/iso-s)']]
    shapley_df['Baseline Deviation from Shapley (%)'] = (shapley_df['Baseline Attribution (gCO2eq/iso-s)'] - shapley_df['Shapley Attribution (gCO2eq/iso-s)']) / shapley_df['Shapley Attribution (gCO2eq/iso-s)'] * 100
    shapley_df['Adjusted Deviation from Shapley (%)'] = (shapley_df['Attribution Adjusted (gCO2eq/iso-s)'] - shapley_df['Shapley Attribution (gCO2eq/iso-s)']) / shapley_df['Shapley Attribution (gCO2eq/iso-s)'] * 100

    shapley_df['Abs Baseline Deviation from Shapley (%)'] = shapley_df['Baseline Deviation from Shapley (%)'].abs()
    shapley_df['Abs Adjusted Deviation from Shapley (%)'] = shapley_df['Adjusted Deviation from Shapley (%)'].abs()

    adjusted_deviation_from_shapley = shapley_df['Abs Adjusted Deviation from Shapley (%)'].mean()
    baseline_deviation_from_shapley = shapley_df['Abs Baseline Deviation from Shapley (%)'].mean()
    worst_case_baseline_deviation = shapley_df['Abs Baseline Deviation from Shapley (%)'].max()
    worst_case_adjusted_deviation = shapley_df['Abs Adjusted Deviation from Shapley (%)'].max()

    df_vals_sample = pd.DataFrame(columns=['grid_ci', 'n_workloads', 'workload', 'partner_workload', 'num_samples', 'baseline_deviation (%)', 'adjusted_deviation (%)'])
    for index, row in shapley_df.iterrows():
        df_vals_sample = df_vals_sample._append({'grid_ci': grid_ci, 'n_workloads': len(workloads_labels), 'workload': index.split('_')[0], 'partner_workload': row['partner_workload'].split('_')[0], 'num_samples': num_samples, 'baseline_deviation (%)': row['Baseline Deviation from Shapley (%)'], 'adjusted_deviation (%)': row['Adjusted Deviation from Shapley (%)']}, ignore_index=True)

    # Print Deviations
    if verbose:
        print('\n')
        print(f'Baseline Deviation from Shapley: {baseline_deviation_from_shapley}')
        print(f'Adjusted Deviation from Shapley: {adjusted_deviation_from_shapley}')
        print(f'Worst Case Baseline Deviation: {worst_case_baseline_deviation}')
        print(f'Worst Case Adjusted Deviation: {worst_case_adjusted_deviation}')

    return adjusted_deviation_from_shapley, baseline_deviation_from_shapley, worst_case_baseline_deviation, worst_case_adjusted_deviation, df_vals_sample


def run_simulation(args):
    min_grid_ci = args[0]
    max_grid_ci = args[1]
    min_workloads = args[2]
    max_workloads = args[3]
    min_samples = args[4]
    max_samples = args[5]
    attribution_workload_candidates = args[6]
    num_trials = args[7]
    data_dir = args[8]
    
    df = pd.DataFrame(columns=['grid_ci', 'n_workloads', 'num_samples', 'adjusted_deviation_from_shapley (%)', 'baseline_deviation_from_shapley (%)', 'worst_case_baseline_deviation (%)', 'worst_case_adjusted_deviation (%)'])
    df_vals = pd.DataFrame(columns=['grid_ci', 'n_workloads', 'workload', 'partner_workload', 'num_samples', 'baseline_deviation (%)', 'adjusted_deviation (%)'])
    
    lifetime = 4 * 365 * 24 * 60 * 60# seconds
    imec_cpu_chip_cf = 20540 # gCO2eq
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

    cpu_chip_cf_per_cpu = imec_cpu_chip_cf
    cpu_cf = cpu_chip_cf_per_cpu * num_cpus_per_node # gCO2eq
    
    cpu_and_cooling_cf = cpu_cf + cooling_cf # gCO2eq

    other_cf = mb_cf + chassis_cf + peripheral_cf + psu_cf + ssd_cf # gCO2eq

    mem_ci = (dram_cf + 0.5 * other_cf) / gb_per_node / lifetime # gCO2eq/(GB-hour)
    cpu_ci = (cpu_and_cooling_cf + 0.5 * other_cf) / (num_cores_per_cpu * num_cpus_per_node) / lifetime # gCO2eq/(core-hour)
    for trial in range(0, num_trials):
        grid_ci = random.randint(min_grid_ci, max_grid_ci)
        n_workloads = random.randint(min_workloads, max_workloads)
        num_samples = random.randint(min_samples, max_samples)
        cf_colocation_matrix = gen_cf_colocation_matrix(data_dir, attribution_workload_candidates, grid_ci, cpu_ci, mem_ci)
        runtime_colocation_matrix = pd.read_csv(f'{data_dir}/runtime_colocation_matrix.csv')
        iso_runtime_colocation_matrix = gen_iso_runtime_normalized_matrix(cf_colocation_matrix, runtime_colocation_matrix)
        attribution_workload_list = gen_colocation(attribution_workload_candidates, n_workloads)
        shapley_df = ground_truth_shapley(data_dir, iso_runtime_colocation_matrix, attribution_workload_list)
        adjusted_deviation_from_shapley, baseline_deviation_from_shapley, worst_case_baseline_deviation, worst_case_adjusted_deviation, df_vals_sample = interference_adjustment(data_dir, shapley_df, grid_ci, num_samples)
        df = df._append({'grid_ci': grid_ci, 'n_workloads': n_workloads, 'num_samples': num_samples, 
                          'baseline_deviation_from_shapley (%)': baseline_deviation_from_shapley, 'adjusted_deviation_from_shapley (%)': adjusted_deviation_from_shapley,
                         'worst_case_baseline_deviation (%)': worst_case_baseline_deviation, 'worst_case_adjusted_deviation (%)': worst_case_adjusted_deviation}, ignore_index=True)
        df_vals = pd.concat([df_vals, df_vals_sample], ignore_index=True)
    return df, df_vals

if __name__ == '__main__':
    # Make sim-results directory
    fair_co2_path = os.environ.get('FAIR_CO2')
    sim_results_dir = f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results'
    os.makedirs(sim_results_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials")
    parser.add_argument("--min_workloads")
    parser.add_argument("--max_workloads")
    parser.add_argument("--min_grid_ci")
    parser.add_argument("--max_grid_ci")
    parser.add_argument("--min_samples")
    parser.add_argument("--max_samples")
    parser.add_argument("--num_workers")
    args = parser.parse_args()
    num_trials = int(args.trials)
    min_workloads = int(args.min_workloads)
    max_workloads = int(args.max_workloads)
    min_grid_ci = int(args.min_grid_ci)
    max_grid_ci = int(args.max_grid_ci)
    min_samples = int(args.min_samples)
    max_samples = int(args.max_samples)
    num_workers = int(args.num_workers)
    attribution_workload_candidates = [
                'removeDuplicates',
                'breadthFirstSearch',
                'minSpanningForest',
                'wordCounts',
                'suffixArray',
                'convexHull',
                'nearestNeighbors',
                'nBody',
                'pgbench-100',
                'pgbench-50',
                'pgbench-10',
                'x265',
                'llama',
                'faiss',
                'spark',
            ] 

    data_dir = f'{fair_co2_path}/colocation/results'

    args_list = []

    interval = (max_grid_ci - min_grid_ci) // num_workers
    trials_per_worker = num_trials // num_workers

    for min_grid_ci in range(min_grid_ci, max_grid_ci, interval):
        print(f'Grid CI: {min_grid_ci} to {min_grid_ci + interval}')
        max_grid_ci = min_grid_ci + interval
        args = (min_grid_ci, max_grid_ci, min_workloads, max_workloads, min_samples, max_samples, attribution_workload_candidates, trials_per_worker, data_dir)
        args_list.append(args)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        result_dfs = executor.map(run_simulation, args_list)
    summary_result_dfs = []
    individual_result_dfs = []
    for result in result_dfs:
        summary_result_dfs.append(result[0])
        individual_result_dfs.append(result[1])
    df = pd.concat(summary_result_dfs, ignore_index=True)
    df_vals = pd.concat(individual_result_dfs, ignore_index=True)

    df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results.csv', index=False)
    df_vals.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_vals.csv', index=False)