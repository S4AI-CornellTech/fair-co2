# Main script that runs all the processing scripts
import process_pcm
import process_logs
import process_docker_stats
import colocation_matrix
import os
import pandas as pd

if __name__ == '__main__':
    workload_list = [
        'pgbench-100',
        'pgbench-50',
        'pgbench-10',
        'x265',
        'llama',
        'faiss',
        'spark',
        'removeDuplicates',
        'breadthFirstSearch',
        'minSpanningForest',
        'wordCounts',
        'suffixArray',
        'convexHull',
        'nearestNeighbors',
        'nBody',
    ]   
    
    fair_co2_path = os.environ.get('FAIR_CO2')

    data_dir = f'{fair_co2_path}/colocation/results'

    interval = 0.1

    static_cpu_power = 35.49978333333334
    static_dram_power = 2.8108166666666663

    rel_runtime_change_per_composite_metric = 0.008817 # (%/(200 GB/s + % CPU Util))
    cpu_util_weight = 1
    memory_bandwidth_weight = 200

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

    kWh_to_J = 3.6e6 # J/kWh
    mem_ci = (dram_cf + 0.5 * other_cf) / gb_per_node / lifetime # gCO2eq/(GB-second)
    cpu_ci = (cpu_and_cooling_cf + 0.5 * other_cf) / (num_cores_per_cpu * num_cpus_per_node) / lifetime # gCO2eq/(core-second)
    node_ci = node_cf / lifetime # gCO2eq/node-hour

    kWh_per_joule = 1 / 3600000
    #grid_ci_list = [0, 50, 100, 300, 500, 1000, 10000]
    grid_ci_list = [0, 500]

    # Find pairwise combinations of workloads
    workload_pairs = []
    for i in range(len(workload_list)):
        for j in range(i, len(workload_list)):
            workload_pairs.append([workload_list[i], workload_list[j]])

    # Process PCM    
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        pcm_log = f'{data_dir}/{results_dir_str}/pcm.log'
        raw_filename = pcm_log[:pcm_log.find('.log')]
        if os.path.exists(f'{raw_filename}_processed.csv'):
            print('Skipping', pcm_log)
            continue
        else:
            print('Processing', pcm_log)
            print(f'{raw_filename}_processed.csv')
            process_pcm.process_pcm(pcm_log, results_dir_str, f'{raw_filename}_processed.csv', 'clr')
    print('PCM logs processed successfully')

    # Process docker stats
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        docker_log_file = f'{results_dir}/docker.log'
        docker_output_file = f'{results_dir}/docker_processed.csv'
        process_docker_stats.process_docker_logs(docker_log_file, docker_output_file)
    print('Docker stats processed successfully')

    # Process pbbs logs
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.process_pbbs_logs(f'{data_dir}/{results_dir_str}')
    print('PBBS logs processed successfully')

    # Find start and end times
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.find_start_end_time(results_dir, perm)
    print('Start and end times found successfully')

    # Fix pgbench start and end times
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.fix_pgbench_start_end_time(results_dir, perm)
    print('Start and end times found successfully')

    # Find average CPU utilization
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.find_avg_cpu_util(results_dir, perm)
    print('Average CPU utilization found successfully')

    # Find average runtime
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.find_avg_runtime(results_dir, perm)
    print('Average runtime found successfully')

    # Find average power and energy
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}'
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.find_avg_power_energy(results_dir, static_cpu_power=static_cpu_power, static_dram_power=static_dram_power)
    print('Average power and energy found successfully')

    # Find embodied carbon footprint
    for perm in (workload_list + workload_pairs):
        results_dir_str = ''
        if len(perm) == 2:
            results_dir_str = f'{perm[0]}_{perm[1]}' 
        else:
            results_dir_str = perm
        results_dir = f'{data_dir}/{results_dir_str}'
        print('Processing:', results_dir)
        process_logs.find_embodied_cf(results_dir, cpu_ci, mem_ci)
    print('Embodied carbon footprint found successfully')

    # Generate new colocation matrix for runtime
    out_file = f'{data_dir}/runtime_colocation_matrix.csv'
    metric = 'runtime'
    colocation_matrix.gen_colocation_matrix(data_dir, workload_list, out_file, metric)
    # Generate new relative change matrix for runtime
    runtime_colocation_matrix_file = f'{data_dir}/runtime_colocation_matrix.csv'
    runtime_relative_change_matrix_file = f'{data_dir}/runtime_relative_change_matrix.csv'
    runtime_relative_change_matrix = colocation_matrix.gen_relative_change_matrix(runtime_colocation_matrix_file, runtime_relative_change_matrix_file)

    # Generate new colocation matrix for proportional energy
    out_file = f'{data_dir}/proportional_energy_colocation_matrix.csv'
    metric = 'proportional total energy (J)'
    colocation_matrix.gen_colocation_matrix(data_dir, workload_list, out_file, metric)
    # Generate new relative change matrix for proportional energy
    energy_colocation_matrix_file = f'{data_dir}/proportional_energy_colocation_matrix.csv'
    energy_relative_change_matrix_file = f'{data_dir}/proportional_energy_relative_change_matrix.csv'
    colocation_matrix.gen_relative_change_matrix(energy_colocation_matrix_file, energy_relative_change_matrix_file)\
    
    # Generate new colocation matrix for proportional dynamic energy
    out_file = f'{data_dir}/proportional_dynamic_energy_colocation_matrix.csv'
    metric = 'proportional dynamic energy (J)'
    colocation_matrix.gen_colocation_matrix(data_dir, workload_list, out_file, metric)
    # Generate new relative change matrix for proportional dynamic energy
    dynamic_energy_colocation_matrix_file = f'{data_dir}/proportional_dynamic_energy_colocation_matrix.csv'
    dynamic_energy_relative_change_matrix_file = f'{data_dir}/proportional_dynamic_energy_relative_change_matrix.csv'
    colocation_matrix.gen_relative_change_matrix(dynamic_energy_colocation_matrix_file, dynamic_energy_relative_change_matrix_file)

    # Generate new colocation matrix for proportional static energy
    out_file = f'{data_dir}/proportional_static_energy_colocation_matrix.csv'
    metric = 'proportional static energy (J)'
    colocation_matrix.gen_colocation_matrix(data_dir, workload_list, out_file, metric)
    # Generate new relative change matrix for proportional static energy
    static_energy_colocation_matrix_file = f'{data_dir}/proportional_static_energy_colocation_matrix.csv'
    static_energy_relative_change_matrix_file = f'{data_dir}/proportional_static_energy_relative_change_matrix.csv'
    colocation_matrix.gen_relative_change_matrix(static_energy_colocation_matrix_file, static_energy_relative_change_matrix_file)

    # Generate new colocation matrix for total power
    out_file = f'{data_dir}/total_power_colocation_matrix.csv'
    metric = 'total power (W)'
    colocation_matrix.gen_colocation_matrix(data_dir, workload_list, out_file, metric)

    # Generate new relative change matrix for total power
    power_colocation_matrix_file = f'{data_dir}/total_power_colocation_matrix.csv'
    power_relative_change_matrix_file = f'{data_dir}/total_power_relative_change_matrix.csv'
    colocation_matrix.gen_relative_change_matrix(power_colocation_matrix_file, power_relative_change_matrix_file)

    # Generate embodied carbon colocation matrix
    ci = cpu_ci * 48 + mem_ci * 96
    runtime_colocation_matrix = pd.read_csv(runtime_colocation_matrix_file)
    embodied_cf_colocation_matrix_file = f'{data_dir}/embodied_cf_colocation_matrix.csv'
    embodied_cf_colocation_matrix = runtime_colocation_matrix.copy()
    # Multiply every element in the matrix by the ci
    embodied_cf_colocation_matrix.iloc[:, 1:] = embodied_cf_colocation_matrix.iloc[:, 1:] * ci
    # Double the values in 'nothing' column
    embodied_cf_colocation_matrix['nothing'] = embodied_cf_colocation_matrix['nothing'] * 2
    embodied_cf_colocation_matrix.to_csv(embodied_cf_colocation_matrix_file, index=False)

    # Find iso runtime normalized embodied carbon sum, static energy sum, and dynamic energy sum
    iso_runtime_normalized_embodied_cf_matrix_file = f'{data_dir}/iso_runtime_normalized_embodied_cf_matrix.csv'
    iso_runtime_normalized_static_energy_matrix_file = f'{data_dir}/iso_runtime_normalized_static_energy_matrix.csv'
    iso_runtime_normalized_energy_matrix_file = f'{data_dir}/iso_runtime_normalized_energy_matrix.csv'
    colocation_matrix.gen_iso_runtime_normalized_matrix(embodied_cf_colocation_matrix_file, runtime_colocation_matrix_file, iso_runtime_normalized_embodied_cf_matrix_file)
    colocation_matrix.gen_iso_runtime_normalized_matrix(static_energy_colocation_matrix_file, runtime_colocation_matrix_file, iso_runtime_normalized_static_energy_matrix_file)
    colocation_matrix.gen_iso_runtime_normalized_matrix(energy_colocation_matrix_file, runtime_colocation_matrix_file, iso_runtime_normalized_energy_matrix_file)

    print('Colocation data processed.')