import subprocess
import local_infra_lib as local
import time
from multiprocessing import Process
import os
import argparse

fair_co2_path = os.environ.get('FAIR_CO2')

def run_spark_workload(workload_label, cpus, memory, results_dir, start_time, rounds):
    interval = 1
    docker_stats_proc = Process(target=local.docker_stats, args=[interval, f'{results_dir}/docker/docker_{workload_label}_{memory}g_{cpus}.log'])
    docker_stats_proc.start()
    pcm_pid = local.start_pcm(interval, f'{results_dir}/pcm/{workload_label}_{memory}g_{cpus}.log')
    time.sleep(20)
    subprocess.run(f'docker exec -i {workload_label} bin/spark-submit --properties-file spark-defaults.conf --master local[${cpus}]  --driver-memory ${memory}G  pyspark_benchmark.py --data_path data/store_sales.csv --start_time {start_time} --rounds {rounds}', shell=True)
    subprocess.run(f'docker cp {workload_label}:/home/spark.txt {results_dir}/spark/spark_{memory}g_{cpus}.log', shell=True)
    subprocess.run(f'docker stop {workload_label}', shell=True)
    local.kill_pid(pcm_pid)
    docker_stats_proc.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Spark workload optimization sweep')
    parser.add_argument('--small', action='store_true', help='Reduce the number of experiment samples')
    parser.add_argument('--medium', action='store_true', help='Reduce the number of experiment samples')
    args = parser.parse_args()
    if args.small:
        cpus_list = [96, 48, 8]
        memory_list = [64, 32, 4]
    elif args.medium:
        cpus_list = [96, 64, 32, 16]
        memory_list = [64, 48, 32, 24]
    else:
        cpus_list = [96, 80, 64, 48, 32, 16, 8]
        memory_list = [64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 2, 1]

    workload_label = 'spark'
    results_dir = f'{fair_co2_path}/workload-optimization/results/spark'
    os.makedirs(f'{results_dir}/docker', exist_ok=True)
    os.makedirs(f'{results_dir}/pcm', exist_ok=True)
    os.makedirs(f'{results_dir}/spark', exist_ok=True)
    rounds = 2
    start_time = 0
    for cpus in cpus_list:
        for memory in memory_list:
            # Initialize the spark container
            subprocess.run(f'docker run -itd --cpus={cpus} --memory={memory}G --memory-swap=-1 --rm=true --name {workload_label} --privileged --net=host --volume {fair_co2_path}/workloads/spark/data:/home/data spark', shell=True)
            run_spark_workload(workload_label, cpus, memory, results_dir, start_time, rounds)
    
