import subprocess
import datetime
import time

def start_pcm(interval, output):
    pcm = subprocess.run(f'sudo nohup pcm {interval} -csv={output} > /dev/null 2>&1 & echo $!', shell=True, capture_output=True)
    return pcm.stdout.decode('utf-8').strip()

def docker_stats(interval, output):
    while(True):
        with open(output, 'a') as f:
            f.write(f'{datetime.datetime.now()}\n')
        subprocess.run(f'docker stats --no-stream >> {output}', shell=True)
        time.sleep(interval)

def stop_docker():
    subprocess.run('docker stop $(docker ps -a -q)')
    subprocess.run('docker rm $(docker ps -a -q)')

def stop_docker_container(container):
    subprocess.run(f'docker stop {container}')
    subprocess.run(f'docker rm {container}')


# Dump detached docker container logs to file for retrieval 
def dump_docker_log(container, file):
    subprocess.run(f'docker logs {container} > {file}')

def start_top(output_file):
    top = subprocess.run(f'nohup top -b > {output_file} 2>&1 & echo $!', shell=True, capture_output=True)
    return top.stdout.decode('utf-8').strip()

def kill_pid(pid):
    subprocess.run(f'sudo kill {pid}', shell=True)

def pkill_process(process):
    subprocess.run(f'sudo pkill {process}', shell=True)