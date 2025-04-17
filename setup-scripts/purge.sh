read -p "Are you sure you want to purge files to make space for FAISS indices? You will not be able to rerun other experiments after this. (y/n) " -n 1 -r
echo 
if [[ $REPLY =~ ^[y]$ ]]
then
    sudo rm -rf $FAIR_CO2/colocation $FAIR_CO2/forecast $FAIR_CO2/monte-carlo-simulations $FAIR_CO2/workloads/spark $FAIR_CO2/workloads/pbbs $FAIR_CO2/x265 $FAIR_CO2/llama
    sudo rm -rf $FAIR_CO2/workload-optimization/results/pbbs/*/pcm $FAIR_CO2/workload-optimization/results/pbbs/*/docker $FAIR_CO2/workload-optimization/results/pbbs/*/pcm_processed $FAIR_CO2/workload-optimization/results/pbbs/*/docker_processed
    sudo rm -rf $FAIR_CO2/workload-optimization/results/spark/pcm $FAIR_CO2/workload-optimization/results/spark/docker $FAIR_CO2/workload-optimization/results/spark/pcm_processed $FAIR_CO2/workload-optimization/results/spark/docker_processed
    sudo rm -rf $HOME/llama.cpp $HOME/tpcds-kit
    docker rmi -f pbbs llama spark x265
fi
