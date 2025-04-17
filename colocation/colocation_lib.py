import subprocess
import  time
from multiprocessing import Process
import datetime
import os

fair_co2_path = os.environ.get('FAIR_CO2')

def run_pbbs_workload(cpus, memory, start_time, workload_name, workload_label, rounds, results_dir):
    subprocess.run(f'docker run --name pbbs_{workload_label} --privileged --net=host\
                    --memory={memory}G --memory-swap=-1 --rm=true\
                    -v {fair_co2_path}/workloads/pbbs/testData:/pbbsbench/testData pbbs\
                    ./runall -maxcpus {cpus} -rounds {rounds} -only {workload_name} -ext -nocheck -start {start_time}\
                    > {results_dir}/pbbs_{workload_label}_{cpus}_{memory}.log', shell=True)


def run_x265_workload(workload_label, rounds, results_dir, start_time):
    # 2 round warm up
    subprocess.run(f'docker exec {workload_label} x265 --input /x265/data/Bosphorus_3840x2160.y4m --output out --pools "+,+" --pmode --pme --wpp --frame-threads 16 --slices 4', shell=True)
    subprocess.run(f'docker exec {workload_label} x265 --input /x265/data/Bosphorus_3840x2160.y4m --output out --pools "+,+" --pmode --pme --wpp --frame-threads 16 --slices 4', shell=True)
    curr_time = time.time()
    sleep_time = start_time - curr_time
    if sleep_time > 0:
        time.sleep(sleep_time)
    round_times = []
    for round in range(0,rounds,1):
        round_start_time = datetime.datetime.now()
        subprocess.run(f'docker exec {workload_label} x265 --input /x265/data/Bosphorus_3840x2160.y4m --output out --pools "+,+" --pmode --pme --wpp --frame-threads 16 --slices 4', shell=True)
        round_end_time = datetime.datetime.now()
        round_time = (round_end_time - round_start_time).total_seconds()
        round_times.append([round, round_start_time, round_end_time, round_time])
    subprocess.run(f'docker stop {workload_label}', shell=True)
    # Save the round times to a csv
    with open(f'{results_dir}/x265.csv', 'w') as f:
        f.write('round,start_time,end_time,time\n')
        for round in round_times:
            f.write(f'{round[0]},{round[1]},{round[2]},{round[3]}\n')
    
def run_pgbench_workload(workload_label, seconds, clients, results_dir, start_time):
    print("Warming up pgbench")
    # 100 second warm up
    subprocess.run(f'docker exec -it --user=postgres {workload_label} pgbench -j 6 -T 100 -c 100 -r --builtin=select postgres', shell=True)
    print("pgbench warmed up")
    curr_time = time.time()
    sleep_time = start_time - curr_time
    if sleep_time > 0:
        time.sleep(sleep_time)
    # Write start time to file
    with open(f'{results_dir}/start', 'w') as f:
        f.write('time\n')
        f.write(f'{datetime.datetime.now()}\n')
    subprocess.run(f'docker exec -it --user=postgres {workload_label} pgbench -j 6 -T {seconds} -c {clients} -r --builtin=select postgres > {results_dir}/{workload_label}.txt', shell=True)

def run_llama_workload(workload_label, cpus, memory, results_dir, batch_size, rounds, model_dir, prompt_size, gen_size, start_time):
    # 2 round warm up
    subprocess.run(f'docker run -it --name {workload_label} --privileged --net=host --memory={memory}G --cpus={cpus} --memory-swap=-1 --rm=true -v $HOME/llama.cpp:$HOME/llama.cpp llama ./build/bin/llama-bench -m models/{model_dir} -o json -t {cpus} -b {batch_size} -pg {prompt_size},{gen_size} -p 0 -n 0 -r 2', shell=True)
    curr_time = time.time()
    sleep_time = start_time - curr_time
    if sleep_time > 0:
        time.sleep(sleep_time)
    subprocess.run(f'docker run -it --name {workload_label} --privileged --net=host --memory={memory}G --cpus={cpus} --memory-swap=-1 --rm=true -v $HOME/llama.cpp:$HOME/llama.cpp llama ./build/bin/llama-bench -m models/{model_dir} -o json -t {cpus} -b {batch_size} -pg {prompt_size},{gen_size} -p 0 -n 0 -r {rounds} > {results_dir}/llama.txt', shell=True)

def run_faiss_workload(workload_label, cpus, memory, results_dir, start_time, rounds):
    subprocess.run(f'docker run -it --name {workload_label} --privileged --net=host --cpus={cpus} --memory={memory}G --rm=true \
                   -v {fair_co2_path}/workloads/faiss/indices:/rag-carbon/faiss_indices \
                   -v {results_dir}:/rag-carbon/data \
                   faiss --start_time {start_time} --rounds {rounds} > {results_dir}/faiss.txt', shell=True)
    
def run_spark_workload(workload_label, cpus, memory, results_dir, start_time, rounds):
    subprocess.run(f'docker exec -i {workload_label} bin/spark-submit --properties-file spark-defaults.conf --master local[${cpus}]  --driver-memory ${memory}G  pyspark_benchmark.py --data_path data/store_sales.csv --start_time {start_time} --rounds {rounds}', shell=True)
    subprocess.run(f'docker cp {workload_label}:/home/spark.txt {results_dir}', shell=True)
    subprocess.run(f'docker stop {workload_label}', shell=True)

def run_schedule(workload_list, results_dir):
    procs = []
    
    if len(workload_list) == 1:
        delay = 0
    else:
        delay = max([workload['min_delay'] for workload in workload_list])
    
    start_offset = min([workload['start'] for workload in workload_list])

    now = int(time.time())

    for workload in workload_list:
        if is_pbbs(workload):
            cpus = workload['cpus']
            memory = workload['memory']
            rounds = workload['rounds']
            workload_name = workload['name']
            workload_label = workload['label']
            start_time = workload['start'] - start_offset + delay + now
            p = Process(target=run_pbbs_workload, args=(cpus, memory, start_time, workload_name, workload_label, rounds, results_dir))
            procs.append(p)
        elif workload['name'] == 'faiss':
            cpus = workload['cpus']
            memory = workload['memory']
            workload_label = workload['label']
            rounds = workload['rounds']
            start_time = workload['start'] - start_offset + delay + now
            p = Process(target=run_faiss_workload, args=(workload_label, cpus, memory, results_dir, start_time, rounds))
            procs.append(p)
        elif workload['name'] == 'llama':
            workload_label = workload['label']
            cpus = workload['cpus']
            memory = workload['memory']
            rounds = workload['rounds']
            model_dir = workload['model_dir']
            batch_size = workload['batch_size']
            prompt_size = workload['prompt_size']
            gen_size = workload['gen_size']
            start_time = workload['start'] - start_offset + delay + now
            p = Process(target=run_llama_workload, args=(workload_label, cpus, memory, results_dir, batch_size, rounds, model_dir, prompt_size, gen_size, start_time))
            procs.append(p)
        elif is_pgbench(workload):
            workload_label = workload['label']
            cpus = workload['cpus']
            memory = workload['memory']
            seconds = workload['seconds']
            clients = workload['clients']
            start_time = workload['start'] - start_offset + delay + now
            p = Process(target=run_pgbench_workload, args=(workload_label, seconds, clients, results_dir, start_time))
            procs.append(p)
        elif workload['name'] == 'x265':
            workload_label = workload['label']
            cpus = workload['cpus']
            memory = workload['memory']
            rounds = workload['rounds']
            # Initialize the x265 container
            subprocess.run(f'docker run -itd --name {workload_label} --privileged --net=host --memory={memory}G --cpus={cpus} --memory-swap=-1 --rm=true -v {fair_co2_path}/workloads/x265:/x265 x265', shell=True)
            start_time = workload['start'] - start_offset + delay + now
            p = Process(target=run_x265_workload, args=(workload_label, rounds, results_dir, start_time))
            procs.append(p)
        elif workload['name'] == 'spark':
            cpus = workload['cpus']
            memory = workload['memory']
            workload_label = workload['label']
            rounds = workload['rounds']
            start_time = workload['start'] - start_offset + delay + now
            # Initialize the spark container
            subprocess.run(f'docker run -itd --cpus={cpus} --memory={memory}G --memory-swap=-1 --rm=true --name {workload_label} --privileged --net=host --volume {fair_co2_path}/workloads/spark/data:/home/data spark', shell=True)
            p = Process(target=run_spark_workload, args=(workload_label, cpus, memory, results_dir, start_time, rounds))
            procs.append(p)
    
    for p in procs:
        p.start()

    for p in procs:
        p.join()
    

def is_pbbs(workload):
    pbbs_workloads = {'removeDuplicates/parlayhash', 'breadthFirstSearch/backForwardBFS', 'minSpanningForest/parallelFilterKruskal', 'wordCounts/histogram', 'suffixArray/parallelKS', 'convexHull/quickHull', 'nearestNeighbors/octTree', 'nBody/parallelCK'}
    name = workload['name']
    return name in pbbs_workloads

def is_pgbench(workload):
    pgbench_workloads = {'pgbench-100', 'pgbench-50', 'pgbench-10'}
    return workload['name'] in pgbench_workloads

def get_workload_name(workload):
    name = workload['name']
    if is_pbbs(workload):
        name = name.split('/')[0]
    return name