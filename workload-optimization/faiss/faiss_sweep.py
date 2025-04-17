import subprocess
import local_infra_lib as local
import time
from multiprocessing import Process
import os
import argparse

fair_co2_path = os.environ.get('FAIR_CO2')

def run_faiss_sweep(interval, cpus):
    results_dir = f'{fair_co2_path}/workload-optimization/results/faiss'
    os.makedirs(results_dir, exist_ok=True)
    docker_stats_proc = Process(target=local.docker_stats, args=[interval, f'{fair_co2_path}/workload-optimization/results/faiss/docker_{cpus}.log'])
    docker_stats_proc.start()
    pcm_pid = local.start_pcm(interval, f'{fair_co2_path}/workload-optimization/results/faiss/pcm_{cpus}.log')
    subprocess.run(f'docker run -it --name faiss --privileged --net=host --cpus={cpus} --rm=true \
                   -v {fair_co2_path}/workloads/faiss/indices:/rag-carbon/faiss_indices \
                   -v {fair_co2_path}/workload-optimization/results/faiss/:/rag-carbon/data \
                   faiss --start_time 0 --rounds 500 --sweep', shell=True)
    local.kill_pid(pcm_pid)
    docker_stats_proc.terminate()
    time.sleep(5)
    # Rename files
    subprocess.run(f'mv {fair_co2_path}/workload-optimization/results/faiss/faiss_times_ivf.csv {fair_co2_path}/workload-optimization/results/faiss/times_{cpus}_ivf.csv', shell=True)
    subprocess.run(f'mv {fair_co2_path}/workload-optimization/results/faiss/faiss_times_hnsw.csv {fair_co2_path}/workload-optimization/results/faiss/times_{cpus}_hnsw.csv', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FAISS workload optimization sweep')
    parser.add_argument('--small', action='store_true', help='Reduce the number of experiment samples')
    parser.add_argument('--medium', action='store_true', help='Reduce the number of experiment samples')
    args = parser.parse_args()
    interval = 0.1
    if args.small:
        cpu_list = [96, 64, 32]
    elif args.medium:
        cpu_list = [96, 80, 64, 48, 32, 16]
    else:
        cpu_list = [96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8]
    for cpus in cpu_list:
        run_faiss_sweep(interval=interval, cpus=cpus)