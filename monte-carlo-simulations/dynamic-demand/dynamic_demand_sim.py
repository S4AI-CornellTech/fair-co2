import random
import pandas as pd
import numpy as np
import itertools
import math
import concurrent.futures
import os
import argparse
from emb_shapley_lib import shapley_attribution as h_shapley

# Schedule is defined as a time series of time slices
# Each time slice contains the timestamp t, the workloads running at that time, and the number of workloads running at that time
# {'t': 0, 'n_wl': 2, 'workloads': [{'workload': 'nBody', 'label': 1, 'start': 0, 'runtime': 10, 'CPU': 48, 'memory': 96}, {'workload': 'nBody', 'label': 2, 'start': 0, 'runtime': 10, 'CPU': 48, 'memory': 96}]}
# Workload list is a list of all workloads with their workload, label, start, iso-runtime, CPU, memory
# {'workload': 'nBody', 'label': 1, 'start': 0, 'runtime': 10, 'CPU': 48, 'memory': 96}

combinations = []
fair_co2_path = os.environ.get('FAIR_CO2')

def choose_workload(candidate_workloads, previous_wl, t, dt, wl_label, workloads, max_time=3, repeat_prob=0.5):
    # Randomly generate a float between 0 and 1
    random_float = random.uniform(0,1)
    
    if previous_wl is None:
        curr_wl = random.choice(candidate_workloads).copy()
        curr_wl['start'] = t
        curr_wl['runtime'] = dt
        curr_wl['label'] = wl_label
        workloads.append(curr_wl)
        wl_label += 1
        return curr_wl, wl_label, workloads
    elif previous_wl['runtime'] >= max_time:
        curr_wl = random.choice(candidate_workloads).copy()
        curr_wl['start'] = t
        curr_wl['runtime'] = dt
        curr_wl['label'] = wl_label
        workloads.append(curr_wl)
        wl_label += 1
        return curr_wl, wl_label, workloads
    elif random_float < repeat_prob:
        # Return previous workload with probability repeat_prob
        curr_wl = previous_wl
        curr_wl['runtime'] += dt
        workloads[previous_wl['label']] = curr_wl
        return curr_wl, wl_label, workloads
    else:
        curr_wl = random.choice(candidate_workloads).copy()
        curr_wl['start'] = t
        curr_wl['runtime'] = dt
        curr_wl['label'] = wl_label
        workloads.append(curr_wl)
        wl_label += 1
        return curr_wl, wl_label, workloads

def gen_schedule(candidate_workloads, time=100, dt=10, max_time = 30, demand=[1,3,4,5,4,2,2,5,3,1]):
    schedule = []
    workloads = []
    wl_label = 0
    i = 0
    for t in range(0, time, dt):
        n_wl = demand[i]
        time_slice = {'t': t, 'n_wl': n_wl, 'workloads': []}
        if t == 0:
            previous_time_slice = None
            previous_time_slice_indices = []
        else:
            previous_time_slice = schedule[-1]
            previous_time_slice_indices = list(range(len(previous_time_slice['workloads'])))
        random.shuffle(previous_time_slice_indices)
        for j in range(n_wl):
            if t == 0:
                previous_wl = None
            elif j < len(previous_time_slice['workloads']):
                idx = previous_time_slice_indices[j]
                previous_wl = previous_time_slice['workloads'][idx]
            else:
                previous_wl = None
            curr_wl, wl_label, workloads = choose_workload(candidate_workloads, previous_wl, t, dt, wl_label, workloads, max_time = max_time, repeat_prob=0.5)
            time_slice['workloads'].append(curr_wl)
        schedule.append(time_slice)
        i += 1
    return schedule, workloads

def get_schedule_energy(schedule, node_static_cpu_power, node_static_mem_power, dt):
    static_energy = 0
    dynamic_energy = 0
    num_nodes = find_num_nodes(schedule)
    time = max([time_slice['t'] for time_slice in schedule]) + 1
    static_energy = num_nodes * time * (node_static_cpu_power + node_static_mem_power)
    for time_slice in schedule:
        dynamic_energy += get_colocation_dynamic_power(time_slice) * dt
    return static_energy, dynamic_energy

def find_num_nodes(schedule):
    num_wl_max = 0
    # Find the max number of wls run on any time slice
    for time_slice in schedule:
        num_wl_max = max(num_wl_max, time_slice['n_wl'])
    # Each workload runs on half a node
    num_nodes = num_wl_max / 2
    # Round up to the nearest whole number
    num_nodes = int(np.ceil(num_nodes))
    return num_nodes

def get_colocation_dynamic_power(time_slice):
    workloads = time_slice['workloads']
    n_wl = len(workloads)
    total_dynamic_power = 0
    for i in range(0, len(workloads), 2):
        workload_0 = workloads[i]
        if i + 1 < n_wl:
            workload_1 = workloads[i + 1]
            node_dynamic_power = get_node_dynamic_power(workload_0, workload_1)
            total_dynamic_power += node_dynamic_power
        else:
            node_dynamic_power = get_node_dynamic_power(workload_0, 'nothing')
            total_dynamic_power += node_dynamic_power
    return total_dynamic_power

def get_node_dynamic_power(dynamic_power_colocation_file, workload_0, workload_1):
    colocation_matrix = pd.read_csv(dynamic_power_colocation_file)
    colocation_matrix.set_index('Workload', inplace=True)
    node_dynamic_power = colocation_matrix.at[workload_0, workload_1]
    return node_dynamic_power

def baseline_attribution(workloads_list):
    n_workloads = len(workloads_list)
    baseline_attributions = [0] * n_workloads
    for i in range(0, n_workloads):
        baseline_attributions[i] = workloads_list[i]['CPU'] * workloads_list[i]['runtime']
    baseline_attributions_sum = sum(baseline_attributions)
    baseline_attributions = [x / baseline_attributions_sum for x in baseline_attributions]
    return baseline_attributions


def get_demand_from_schedule(workload_list, time, dt):
    demand = [0] * (time//dt)
    i = 0
    for t in range(0, time, dt):
        n_cpus = 0
        for workload in workload_list:
            if workload['start'] <= t and workload['start'] + workload['runtime'] > t:
                n_cpus += workload['CPU']
        demand[i] = n_cpus
        i += 1
    return demand

def get_shapley_attribution(args):
    workload_label = args[0]
    time = args[1]
    dt = args[2]
    workload_list = args[3]
    num_workloads = len(workload_list)

    shapley_value = 0
    
    for combination in combinations:
        if workload_label in combination:
            combination_without_workload = combination - {workload_label}
            wl_combination_without_workload = [wl for wl in workload_list if wl['label'] in combination_without_workload]
            wl_combination = [wl for wl in workload_list if wl['label'] in combination]
            demand_without_workload = get_demand_from_schedule(wl_combination_without_workload, time, dt)
            demand_with_workload = get_demand_from_schedule(wl_combination, time, dt)
            peak_without_workload = max(demand_without_workload)
            peak_with_workload = max(demand_with_workload)
            scaling_factor = 1 / math.comb(num_workloads-1, len(combination_without_workload))
            shapley_value += (peak_with_workload - peak_without_workload) * scaling_factor 
    return shapley_value

def ground_truth_shapley_attribution_parallel(workload_list, time, dt, max_workers=4):
    workload_labels = [workload['label'] for workload in workload_list]
    num_workloads = len(workload_list)
    global combinations
    combinations = []
    for k in range(0, num_workloads + 1):
        combinations += list(itertools.combinations(workload_labels, k))
    for i in range(0, len(combinations)):
        combinations[i] = set(combinations[i])
    args_list = [(workload_label, time, dt, workload_list) for workload_label in workload_labels]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        shapley_attributions = executor.map(get_shapley_attribution, args_list)
    shapley_attributions = list(shapley_attributions)
    shapley_attributions_sum = sum(shapley_attributions)
    shapley_attributions = [x / shapley_attributions_sum for x in shapley_attributions]
                                
    return shapley_attributions

def ground_truth_shapley_attribution(workload_list, time, dt):
    workload_labels = [workload['label'] for workload in workload_list]
    num_workloads = len(workload_list)
    for k in range(0, num_workloads + 1):
        combinations += list(itertools.combinations(workload_labels, k))
    for i in range(0, len(combinations)):
        combinations[i] = set(combinations[i])
    shapley_attributions = []

    for workload_label in workload_labels:
        shapley_value = 0
        for combination in combinations:
            if workload_label in combination:
                combination_without_workload = combination - {workload_label}
                wl_combination_without_workload = [wl for wl in workload_list if wl['label'] in combination_without_workload]
                wl_combination = [wl for wl in workload_list if wl['label'] in combination]
                demand_without_workload = get_demand_from_schedule(wl_combination_without_workload, time, dt)
                demand_with_workload = get_demand_from_schedule(wl_combination, time, dt)
                peak_without_workload = max(demand_without_workload)
                peak_with_workload = max(demand_with_workload)
                scaling_factor = 1 / math.comb(num_workloads-1, len(combination_without_workload))
                shapley_value += (peak_with_workload - peak_without_workload) * scaling_factor 
        shapley_attributions.append(shapley_value)
    
    shapley_attributions_sum = sum(shapley_attributions)
    shapley_attributions = [x / shapley_attributions_sum for x in shapley_attributions]
                                
    return shapley_attributions

def temporal_shapley_shapley_attribution(workload_list, time, dt):
    demand = get_demand_from_schedule(workload_list, time, dt)
    time_list = [i for i in range(0, time, dt)]
    demand_df = pd.DataFrame({'time': time_list, 'demand': demand})
    time_granularities = [dt]
    shapley, peaks_per_granularity, ci_list, resource_time_val = h_shapley(demand_df, 'time', 'demand', time_granularities, attribution_total=1, sampling_interval=dt, offset=0)
    ci = ci_list[-1]
    attributions = []
    for workload in workload_list:
        attribution_value = 0
        for i in range(0, time, dt):
            if workload['start'] <= i and workload['start'] + workload['runtime'] > i:
                attribution_value += ci[i] * dt * workload['CPU']
        attributions.append(attribution_value)
    attributions_sum = sum(attributions)
    attributions = [x / attributions_sum for x in attributions]
    return attributions

def demand_proportional(workload_list, time, dt):
    demand = get_demand_from_schedule(workload_list, time, dt)
    ci = [0] * len(demand)
    for i in range(0, len(demand)):
        ci[i] = demand[i] / sum(demand)
    attributions = []
    for workload in workload_list:
        attribution_value = 0
        for i in range(0, time, dt):
            if workload['start'] <= i and workload['start'] + workload['runtime'] > i:
                attribution_value += ci[i] * dt * workload['CPU']
        attributions.append(attribution_value)
    attributions_sum = sum(attributions)
    attributions = [x / attributions_sum for x in attributions]
    return attributions

def h_shap_experiment(batches=1000, num_trials_per_batch=10, max_n_workloads=22, max_time=3, min_time_slices=4, max_time_slices=10, min_demand=1, max_demand=5, max_workers=20):
    candidate_workloads = [{'workload': 'A', 'CPU': 8},
                            {'workload': 'B', 'CPU': 16},
                            {'workload': 'C', 'CPU': 32},
                            {'workload': 'D', 'CPU': 48},
                            {'workload': 'E', 'CPU': 64},
                            {'workload': 'F', 'CPU': 80},
                            {'workload': 'G', 'CPU': 96}]

    demand = [2,3,4,5,4,2,2,3,4,5]
    
    dt = 1
    max_time = max_time * dt
    data_df = pd.DataFrame({'num_workloads': [], 
                            'num_time_slices': [], 
                            'baseline_avg_deviation (%)': [], 
                            'demand_proportional_avg_deviation (%)': [], 
                            'temporal_shapley_avg_deviation (%)': [], 
                            'baseline_worst_case_deviation (%)': [],
                            'demand_proportional_worst_case_deviation (%)': [],
                            'temporal_shapley_worst_case_deviation (%)': []})
    
    data_df_vals = pd.DataFrame({'num_workloads': [],
                                 'num_time_slices': [],
                                 'workload_cpus': [],
                                 'workload_runtime': [],
                                 'baseline_deviation (%)': [],
                                 'demand_proportional_deviation (%)': [],
                                 'temporal_shapley_deviation (%)': []})
    
    data_df = data_df.astype({'num_workloads': 'int32', 
                              'num_time_slices': 'int32', 
                              'baseline_avg_deviation (%)': 'float64', 
                              'demand_proportional_avg_deviation (%)': 'float64', 
                              'temporal_shapley_avg_deviation (%)': 'float64',
                              'baseline_worst_case_deviation (%)': 'float64',
                              'demand_proportional_worst_case_deviation (%)': 'float64',
                              'temporal_shapley_worst_case_deviation (%)': 'float64'})
    for batch in range(0, batches):
        for trial in range(0, num_trials_per_batch):
            print(f'Trial {trial}')
            num_time_slices = np.random.randint(min_time_slices, max_time_slices)
            time = dt * num_time_slices
            demand = [random.randint(min_demand, max_demand) for i in range(0, time, dt)]
            n_workloads = max_n_workloads + 1
            while (n_workloads > max_n_workloads):
                schedule, workloads = gen_schedule(candidate_workloads, time=time, dt=dt, max_time=max_time, demand=demand)
                n_workloads = len(workloads)
            print(f'Generated schedule with {n_workloads} workloads')
            
            for workload in workloads:
                print(workload)
            print('\n')

            
            print("Parallel Shapley attributions:")
            shapley_attributions = ground_truth_shapley_attribution_parallel(workloads, time, dt, max_workers=max_workers)
            for i in range(0, n_workloads):
                print(f'Workload {i} has Shapley value {shapley_attributions[i]}')

            print("\n")

            print("Baseline attributions:")
            baseline_attributions = baseline_attribution(workloads)
            for i in range(0, n_workloads):
                print(f'Workload {i} has baseline value {baseline_attributions[i]}')

            print("\n")

            print("Demand-based baseline attributions:")
            demand_proportional_attributions = demand_proportional(workloads, time, dt)
            for i in range(0, n_workloads):
                print(f'Workload {i} has demand-based baseline value {demand_proportional_attributions[i]}')

            print("\n")

            print("Temporal Shapley attributions:")
            temporal_shapley_attributions = temporal_shapley_shapley_attribution(workloads, time, dt)
            for i in range(0, n_workloads):
                print(f'Workload {i} has Temporal Shapley value {temporal_shapley_attributions[i]}')

            print("\n")

            # Plot deviation from shapley attributions
            shapley_attributions = np.array(shapley_attributions)
            baseline_attributions = np.array(baseline_attributions)
            demand_proportional_attributions = np.array(demand_proportional_attributions)
            temporal_shapley_attributions = np.array(temporal_shapley_attributions)
            deviation_baseline = np.abs((baseline_attributions - shapley_attributions) / shapley_attributions) * 100
            deviation_demand_proportional = np.abs((demand_proportional_attributions - shapley_attributions) / shapley_attributions) * 100
            deviation_temporal_shapley = np.abs((temporal_shapley_attributions - shapley_attributions) / shapley_attributions) * 100

            data_df = data_df._append({'num_workloads': int(n_workloads), 
                                    'num_time_slices': int(num_time_slices),
                                    'baseline_avg_deviation (%)': np.mean(deviation_baseline), 
                                    'demand_proportional_avg_deviation (%)': np.mean(deviation_demand_proportional), 
                                    'temporal_shapley_avg_deviation (%)': np.mean(deviation_temporal_shapley),
                                    'baseline_worst_case_deviation (%)': np.max(deviation_baseline),
                                    'demand_proportional_worst_case_deviation (%)': np.max(deviation_demand_proportional),
                                    'temporal_shapley_worst_case_deviation (%)': np.max(deviation_temporal_shapley),
                                    }, ignore_index=True)
            
            for i, workload in enumerate(workloads):
                data_df_vals = data_df_vals._append({'num_workloads': int(n_workloads),
                                                     'num_time_slices': int(num_time_slices),
                                                     'workload_cpus': workload['CPU'],
                                                     'workload_runtime': workload['runtime'],
                                                     'baseline_deviation (%)': deviation_baseline[i],
                                                     'demand_proportional_deviation (%)': deviation_demand_proportional[i],
                                                     'temporal_shapley_deviation (%)': deviation_temporal_shapley[i]
                                                     }, ignore_index=True)

        print(f'Average deviation from Shapley attributions for baseline: {data_df["baseline_avg_deviation (%)"].mean()}')
        print(f'Average deviation from Shapley attributions for demand-proportional: {data_df["demand_proportional_avg_deviation (%)"].mean()}')
        print(f'Average deviation from Shapley attributions for temporal_shapley: {data_df["temporal_shapley_avg_deviation (%)"].mean()}')
        data_df = data_df.astype({'num_workloads': 'int32', 
                                  'num_time_slices': 'int32', 
                                  'baseline_avg_deviation (%)': 'float64', 
                                  'demand_proportional_avg_deviation (%)': 'float64', 
                                  'temporal_shapley_avg_deviation (%)': 'float64',
                                  'baseline_worst_case_deviation (%)': 'float64',
                                  'demand_proportional_worst_case_deviation (%)': 'float64',
                                  'temporal_shapley_worst_case_deviation (%)': 'float64'})
        data_df_vals = data_df_vals.astype({'num_workloads': 'int32',
                                             'num_time_slices': 'int32',
                                             'workload_cpus': 'int32',
                                             'workload_runtime': 'int32',
                                             'baseline_deviation (%)': 'float64',
                                             'demand_proportional_deviation (%)': 'float64',
                                             'temporal_shapley_deviation (%)': 'float64'})
        # CSV
        data_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule.csv')
        data_df_vals.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_vals.csv')
        # Pickle 
        data_df.to_pickle(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule.pkl')
        data_df_vals.to_pickle(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_vals.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials")
    parser.add_argument("--max_workloads")
    parser.add_argument("--min_time_slices")
    parser.add_argument("--max_time_slices")
    parser.add_argument("--num_workers")
    args = parser.parse_args()
    trials = int(args.trials)
    max_workloads = int(args.max_workloads)
    min_time_slices = int(args.min_time_slices)
    max_time_slices = int(args.max_time_slices)
    num_workers = int(args.num_workers)
    num_trials_per_batch = 10
    batches = trials // num_trials_per_batch
    # Make sim-results directory
    fair_co2_path = os.environ.get('FAIR_CO2')
    sim_results_dir = f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results'
    os.makedirs(sim_results_dir, exist_ok=True)
    h_shap_experiment(batches=batches, num_trials_per_batch=num_trials_per_batch, max_n_workloads=max_workloads, min_time_slices=min_time_slices, max_time_slices=max_time_slices, max_workers=num_workers)



        