import colocation_lib as colocation
from multiprocessing import Process
import local_infra_lib as local
import argparse
import time
import subprocess
import os

if __name__ == '__main__':
    fair_co2_path = os.environ.get('FAIR_CO2')
    data_dir = f'{fair_co2_path}/colocation/results'
    # Make the data directory
    os.makedirs(data_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument('--small', action='store_true', help='Reduce the number of experiment samples')
    args = parser.parse_args()
    model_name = str(args.model)
    if args.small:
        workload_list = [
            {'name': 'pgbench-100', 'cpus': 48, 'memory': 96, 'clients': 100, 'seconds': 250, 'id': 0, 'start': 0, 'min_delay': 120, 'label': 'pgbench-100', 'runtime': 1}, # Good
            {'name': 'x265', 'cpus': 48, 'memory': 96, 'rounds': 6, 'id': 3, 'start': 0, 'min_delay': 150, 'label': 'x265', 'runtime': 50}, # Good
            {'name': 'breadthFirstSearch/backForwardBFS', 'cpus': 48, 'memory': 96, 'rounds': 30, 'id': 8, 'start': 0, 'min_delay': 200, 'label': 'breadthFirstSearch', 'runtime': 7.09}, # 7.09 seconds with 32 cores
        ]
    else:
        workload_list = [
            {'name': 'pgbench-100', 'cpus': 48, 'memory': 96, 'clients': 100, 'seconds': 250, 'id': 0, 'start': 0, 'min_delay': 120, 'label': 'pgbench-100', 'runtime': 1}, # Good
            {'name': 'pgbench-50', 'cpus': 48, 'memory': 96, 'clients': 50, 'seconds': 250, 'id': 1, 'start': 0, 'min_delay': 120, 'label': 'pgbench-50', 'runtime': 1}, # Good
            {'name': 'pgbench-10', 'cpus': 48, 'memory': 96, 'clients': 10, 'seconds': 250, 'id': 2, 'start': 0, 'min_delay': 120, 'label': 'pgbench-10', 'runtime': 1}, # Good
            {'name': 'x265', 'cpus': 48, 'memory': 96, 'rounds': 6, 'id': 3, 'start': 0, 'min_delay': 150, 'label': 'x265', 'runtime': 50}, # Good
            {'name': 'llama', 'cpus': 48, 'memory': 96, 'rounds': 5, 'id': 4, 'start': 0, 'min_delay': 150, 'label': 'llama', 'batch_size': 1, 'prompt_size': 128, 'gen_size': 64, 'model_dir': model_name}, # Good
            {'name': 'faiss', 'cpus': 48, 'memory': 96, 'rounds': 300, 'id': 5, 'start': 0, 'min_delay': 300, 'label': 'faiss', 'runtime': 1}, # Good
            {'name': 'spark', 'cpus': 48, 'memory': 96, 'rounds' : 4, 'id': 6, 'start': 0, 'min_delay': 100, 'label': 'spark', 'runtime' : 75},
            {'name': 'removeDuplicates/parlayhash', 'cpus': 48, 'memory': 96, 'rounds': 30, 'id': 7, 'start': 0, 'min_delay': 100, 'label': 'removeDuplicates', 'runtime': 5.135}, # 5.8 seconds with 32 cores
            {'name': 'breadthFirstSearch/backForwardBFS', 'cpus': 48, 'memory': 96, 'rounds': 30, 'id': 8, 'start': 0, 'min_delay': 200, 'label': 'breadthFirstSearch', 'runtime': 7.09}, # 7.09 seconds with 32 cores
            {'name': 'minSpanningForest/parallelFilterKruskal', 'cpus': 48, 'memory': 155, 'rounds': 30, 'id': 9, 'start': 0, 'min_delay': 240, 'label': 'minSpanningForest', 'runtime': 7.502},# 9 seconds with 32 cores
            {'name': 'wordCounts/histogram', 'cpus': 48, 'memory': 96, 'rounds': 60, 'id': 10, 'start': 0, 'min_delay': 100, 'label': 'wordCounts', 'runtime': 2.46}, # 4 seconds with 32 cores
            {'name': 'suffixArray/parallelKS', 'cpus': 48, 'memory': 96, 'rounds': 8, 'id': 11, 'start': 0, 'min_delay': 100, 'label': 'suffixArray', 'runtime': 35.127}, # 55 seconds with 32 cores
            {'name': 'convexHull/quickHull', 'cpus': 48, 'memory': 96, 'rounds': 8, 'id': 12, 'start': 0, 'min_delay': 200, 'label': 'convexHull', 'runtime': 32.536}, # 50 seconds with 32 cores
            {'name': 'nearestNeighbors/octTree', 'cpus': 48, 'memory': 96, 'rounds': 30, 'id': 13, 'start': 0, 'min_delay': 100, 'label': 'nearestNeighbors', 'runtime': 5.461}, # 10 seconds with 32 cores
            {'name': 'nBody/parallelCK', 'cpus': 48, 'memory': 96, 'rounds': 10, 'id': 14, 'start': 0, 'min_delay': 100, 'label': 'nBody', 'runtime': 18.427}, # 27 seconds with 48 cores
        ]

    interval = 0.1

    # # Run each workload once to generate isolated data
    for workload in workload_list:
        workload_pair = [workload]
        workload_name = workload["name"].split('/')[0]
        workload_label = workload['label']
        results_dir = f'{fair_co2_path}/colocation/results/{workload_name}'
        # Make the results directory
        print('Results dir:', results_dir)
        subprocess.run(f'mkdir -p {results_dir}', shell=True)
        if colocation.is_pgbench(workload):
            print("Initializing pgbench...")
            subprocess.run(f'docker run -itd --cpus=48 --memory=96G --name {workload_label} --rm=true -e POSTGRES_PASSWORD=Welcome4$ -p 5432:5432 postgres', shell=True)
            time.sleep(5)
            subprocess.run(f'docker exec -it --user=postgres {workload_label} pgbench -i -s 1000 postgres', shell=True)
            print("pgbench initialized.")
        docker_stats_proc = Process(target=local.docker_stats, args=[interval, f'{results_dir}/docker.log'])
        docker_stats_proc.start()
        pcm_pid = local.start_pcm(interval, f'{results_dir}/pcm.log')
        time.sleep(20)
        colocation.run_schedule(workload_pair, results_dir)
        local.kill_pid(pcm_pid)
        docker_stats_proc.terminate()
        if colocation.is_pgbench(workload):
            print("Stopping pgbench...")
            subprocess.run(f'docker stop {workload_label}', shell=True)
            print("pgbench stopped.")

    # Find all pairwise combinations of workloads
    for workload in workload_list:
        workload_name = workload["name"].split('/')[0]
        workload_label = workload['label']
        if colocation.is_pgbench(workload):
            print("Initializing pgbench...")
            subprocess.run(f'docker run -itd --cpus=48 --memory=96G --name {workload_label} --rm=true -e POSTGRES_PASSWORD=Welcome4$ -p 5432:5432 postgres', shell=True)
            time.sleep(5)
            subprocess.run(f'docker exec -it --user=postgres {workload_label} pgbench -i -s 1000 postgres', shell=True)
        for workload2 in workload_list:
            workload2_name = workload2["name"].split('/')[0]
            if (workload2['id'] < workload['id']):
                continue
            workload_1_copy = workload.copy()
            workload_2_copy = workload2.copy()
            workload_name = workload['label']
            workload2_name = workload2['label']
            if workload_name == workload2_name:
                workload_2_copy['label'] = f'{workload2_name}_1'
            workload_2_label = workload_2_copy['label']
            workload_pair = [workload_1_copy, workload_2_copy]
            results_dir = f'{fair_co2_path}/colocation/results/{workload_name}_{workload2_name}'
            # Make the results directory
            print('Results dir:', results_dir)
            subprocess.run(f'mkdir -p {results_dir}', shell=True)
            if colocation.is_pgbench(workload2):
                print("Initializing pgbench...")
                subprocess.run(f'docker run -itd --cpus=48 --memory=96G --name {workload_2_label} --rm=true -e POSTGRES_PASSWORD=Welcome4$ -p 5433:5433 postgres', shell=True)
                time.sleep(5)
                subprocess.run(f'docker exec -it --user=postgres {workload_2_label} pgbench -i -s 1000 postgres', shell=True)
                print("pgbench initialized.")
            docker_stats_proc = Process(target=local.docker_stats, args=[interval, f'{results_dir}/docker.log'])
            docker_stats_proc.start()
            pcm_pid = local.start_pcm(interval, f'{results_dir}/pcm.log')
            time.sleep(20)
            colocation.run_schedule(workload_pair, results_dir)
            local.kill_pid(pcm_pid)
            docker_stats_proc.terminate()
            if colocation.is_pgbench(workload2):
                print("Stopping pgbench...")
                subprocess.run(f'docker stop {workload_2_label}', shell=True)
                print("pgbench stopped.")
        if colocation.is_pgbench(workload):
            print("Stopping pgbench...")
            subprocess.run(f'docker stop {workload_label}', shell=True)
            print("pgbench stopped.")