# Build dependencies
sudo apt update
sudo apt install -y libjemalloc2 numactl python3 python3-pip build-essential 

$FAIR_CO2/workloads/pbbs/runall -maxcpus 96 -rounds 1 -only nBody/parallelCK -ext -nocheck -start 0

docker build --network=host -t pbbs .