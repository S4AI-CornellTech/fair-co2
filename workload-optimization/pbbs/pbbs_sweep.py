import subprocess
import local_infra_lib as local
import time
from multiprocessing import Process
import os
import argparse

fair_co2_path = os.environ.get('FAIR_CO2')

def run_pbbs(cpus, memory, workload, interval, rounds, results_dir):
    workload_name = workload['name']
    workload_name_split = workload_name.split('/')[0]
    docker_stats_proc = Process(target=local.docker_stats, args=[interval, f'{results_dir}/docker_{workload_name_split}_{cpus}_{memory}.log'])
    docker_stats_proc.start()
    pcm_pid = local.start_pcm(interval, f'{results_dir}/pcm_{workload_name_split}_{cpus}_{memory}.log')
    time.sleep(20)
    print(f'Running pbbs {workload_name} with {cpus} cpus and {memory}GB memory')
    subprocess.run(f'docker exec -it pbbs  ./runall -maxcpus {cpus} -rounds {rounds} -only {workload_name} -ext -nocheck -start 0\
                    > {results_dir}/pbbs_{workload_name_split}_{cpus}_{memory}.log', shell=True)
    local.kill_pid(pcm_pid)
    docker_stats_proc.terminate()

def run_pbbs_sweep(cpus_list, memory_list, workload_list, interval):
    results_dir = f'{fair_co2_path}/workload-optimization/results/pbbs'
    os.makedirs(results_dir, exist_ok=True)
    for memory in memory_list:
        subprocess.run(f'docker run -itd --name pbbs --privileged --net=host\
                        --memory={memory}G --memory-swap=-1 --rm=true\
                        -v {fair_co2_path}/workloads/pbbs/testData:/pbbsbench/testData pbbs /bin/bash', shell=True)
        for cpus in cpus_list:
            for workload in workload_list:
                if memory < workload['min_memory']:
                    print(f"Skipping {workload['name']} with {cpus} cpus and {memory}GB memory")
                else:
                    rounds = workload['rounds']
                    run_pbbs(cpus, memory, workload, interval, rounds, results_dir)
        subprocess.run('docker stop pbbs', shell=True)
        time.sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PBBS workload optimization sweep')
    parser.add_argument('--small', action='store_true', help='Reduce the number of experiment samples')
    parser.add_argument('--medium', action='store_true', help='Reduce the number of experiment samples')
    parser.add_argument('--workload', type=str, default='all', help='Specify a workload to run from {ddup, bfs, msf, wc, sa, ch, nn, nbody} or use "all" to run all workloads')
    args = parser.parse_args()
    if args.small:
        cpus_list = [96, 64, 32]
        memory_list = [96, 64, 48]
    elif args.medium:
        cpus_list = [96, 64, 32, 16]
        memory_list = [96, 64, 48, 32, 24]
    else:
        cpus_list = [96, 80, 64, 48, 32, 16]
        memory_list = [192, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8]

    # Define the workloads to run
    # If a specific workload is specified, filter the workload_list
    if args.workload == 'ddup':
        workload_list = [{'name': 'removeDuplicates/parlayhash', 'rounds': 10, 'min_memory': 8}]
    elif args.workload == 'bfs':
        workload_list = [{'name': 'breadthFirstSearch/backForwardBFS', 'rounds': 10, 'min_memory': 8}]
    elif args.workload == 'msf':
        workload_list = [{'name': 'minSpanningForest/parallelFilterKruskal', 'rounds': 10, 'min_memory': 48}]
    elif args.workload == 'wc':
        workload_list = [{'name': 'wordCounts/histogram', 'rounds': 10, 'min_memory': 8}]
    elif args.workload == 'sa':
        workload_list = [{'name': 'suffixArray/parallelKS', 'rounds': 3, 'min_memory': 8}]
    elif args.workload == 'ch':
        workload_list = [{'name': 'convexHull/quickHull', 'rounds': 3, 'min_memory': 8}]
    elif args.workload == 'nn':
        workload_list = [{'name': 'nearestNeighbors/octTree', 'rounds': 10, 'min_memory': 24}]
    elif args.workload == 'nbody':
        workload_list = [{'name': 'nBody/parallelCK', 'rounds': 5, 'min_memory': 8}]
    elif args.workload == 'all':
        workload_list = [
            {'name': 'removeDuplicates/parlayhash', 'rounds': 10, 'min_memory': 8}, # 5.8 seconds with 32 cores
            {'name': 'breadthFirstSearch/backForwardBFS', 'rounds': 10, 'min_memory': 8}, # 9.5 seconds with 32 cores
            {'name': 'minSpanningForest/parallelFilterKruskal', 'rounds': 10, 'min_memory': 48},# 9 seconds with 32 cores
            {'name': 'wordCounts/histogram', 'rounds': 10, 'min_memory': 8}, # 4 seconds with 32 cores
            {'name': 'suffixArray/parallelKS', 'rounds': 3, 'min_memory': 8}, # 55 seconds with 32 cores
            {'name': 'convexHull/quickHull', 'rounds': 3, 'min_memory': 8}, # 50 seconds with 32 cores
            {'name': 'nearestNeighbors/octTree', 'rounds': 10, 'min_memory': 24}, # 10 seconds with 32 cores
            {'name': 'nBody/parallelCK', 'rounds': 5, 'min_memory': 8}, # 27 seconds with 32 cores
        ]
    else:
        print(f"Unknown workload: {args.workload}\n Please specify a valid workload from {{ddup, bfs, msf, wc, sa, ch, nn, nbody}} or use 'all' to run all workloads.")
        exit(1)

    interval = 0.1
    run_pbbs_sweep(cpus_list=cpus_list, memory_list=memory_list, workload_list=workload_list, interval=interval)