# Fair-CO2: Fair Attribution for Cloud Data Centers
Artifact repository for Fair-CO2

We recommend running all scripts and experiments in a tmux session as some scripts and experiments can take a long time to run. 

As many of these experiments can take a very long time to run, we provide options (via <code>--small</code> command line argument when noted) to run a smaller subset of the experiments. However, running the smaller experiment set is not enough to reproduce results from the paper and should only be done to check functionality of the experimental code.

#### Using Provided Experimental Data
To enable reproduction of paper results without running all workload experiments on hardware, we provide partially pre-processed experimental data for all experiments. When applicable, we note how to use provided data for data analysis and figure generation.

## 0. Pre-requisites
Fair-CO2 experimental results are collected on the compute_cascadelake_r nodes hosted at [CHI@TACC](https://chi.tacc.chameleoncloud.org) as part of the [Chameleon platform](https://www.chameleoncloud.org/).

Paper test system specifications:
- Number of CPUs: 2
- CPU: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz (24 physical cores, 48 logical cores)
- RAM size: 192 GB
- Storage capacity: 420 GB

Chameleon image: CC-Ubuntu22.04 (ID=f837a6c1-69e8-4ed7-914f-1d3562e90a89)

We recommend testing on a bare-metal system with root access with the following requirements:
- Number of CPUS: 2
- Physical cores per CPU: 24 
- Logical cores per CPU: 48
- RAM size: 192 GiB
- Storage capacity: > 600 GB
- Operating system: Ubuntu 22.04

Note: we recommend a storage capacity greater than the system we tested on because setting up all the experiments will consume more storage than is available on the Chameleon node that was tested.

## 1. Set Up
To set up all workloads, run the following set up scripts in /setup-scripts. To set up only for a specific workload, replace `source setup-scripts/setup_workloads` with `source workloads/<workload>/setup.sh`. If you are planning to run only a specific PBBS workload, run `source workloads/pbbs/setup_<pbbs_workload>.sh` instead where `<pbbs_workload>` can be any one of `{ddup, bfs, msf, wc, sa, ch, nn, nbody}`. 
```
source setup-scripts/env.sh
source setup-scripts/create_large_swap.sh
source setup-scripts/install_pcm.sh
source setup-scripts/install_docker.sh
source setup-scripts/install_conda.sh
source setup-scripts/setup_workloads.sh
source setup-scripts/delete_swap.sh
```

Create and activate the conda environment
```
conda env create -f environment.yml
conda activate fair-co2
```
### Download the Llama Model
We use the following LLAMA 3 8B model from Hugging Face: https://huggingface.co/meta-llama/Meta-Llama-3-8B. To download the model, please follow the access request and download instructions on the Hugging Face model website. Use the llama.cpp utility <code>convert_hf_to_gguf.py</code> to convert the model to a .gguf file. Place the model .gguf file in the ~/llama.cpp/models folder. Alternatively, you may use another .gguf LLM model; just modify the model file name accordingly when running the profiling scripts.

### Generate the IVF FAISS Index
Note: this is a very lengthy step, taking potentially multiple consecutive days
```
python3 workloads/faiss/gen_IVF_SQ8_100M_index.py
```

### Generate the Data for Apache Spark
Note: this step takes around 1 hour
```
source workloads/spark/gen_data.sh
```

## 2. Pairwise Colocation Profiling (Figure 2)
This set of experiments will run all pairwise colocations along with isolated runs of each workload. Each workload is constrained to use only half the resources (48 logical cores and 96 GB of memory) via Docker. Steps 2a and 2b run the colocation experiments and process the raw logs, step 2c generates colocation matrix figures 2a and 2b from the paper. Steps 2a and 2b can take a while so we provide reference pre-processed experiment logs in <code>colocation/ref-results</code>. 

#### Using Provided Experimental Data
To use the provided experimental data for step 2c and step 3, copy the content from <code>colocation/ref-results</code> to <code>colocation/results</code> and skip steps 2a and 2b.
```
mkdir -p colocation/results
cp -r colocation/ref-results/* colocation/results/
```

### 2a. Colocation Experiments
Run the following script on the test server to start the experiment. The results will be stored in <code>colocation/results</code>. The experiments will take a while to run (maybe around one whole day).
```
python3 colocation/colocation_sweep.py --model Meta-Llama-3-8B-F16.gguf
```

To run a small subset of the colocation experiments, use the <code>--small</code> argument. Note that this option will not collect enough data to reproduce the results from the paper and should only be used as a demo of the sweep experiments.
```
python3 colocation/colocation_sweep.py --small --model Meta-Llama-3-8B-F16.gguf
```

### 2b. Processing Colocation Experiment Logs
The colocation experiments will generate workload-specific logs with performance and task-specific information. Each experiment run will also have a <code>docker.log</code> file with outputs from <code>docker stats</code> that detail CPU utilization at a container-granularity. Each run also has a <code>pcm.log</code> that include system-level performance telemetry such as CPU package power and DRAM power.

Depending on the machine, PCM may print a few lines that are similar to "Link 3 is disabled" before returning any data. In order to properly process this, please test `sudo pcm` and observe the number of times this line is printed. Then update line 5 in the file colocation/process_pcm.py to reflect this. Likewise, update line 6 with the number of logical cores on the machine.

Once this is done, run the following script to process the raw logs.
```
python3 colocation/process_colocation_sweep.py
```

### 2c. Generating Colocation Matrix Figures
Run the following script to generate figures 2a and 2b from the paper. The figures will be saved to the <code>figures</code> folder.
```
python3 colocation/gen_colocation_sweep_figures.py
```

## 3. Monte Carlo Simulations (Figures 7, 8, 9)
This set of experiments runs the Monte Carlo simulations of randomly generated dynamic demand/workload schedules and colocation scenarios. For each simulated scenario, the ground truth Shapley value method is used to create a ground truth carbon attribution for each workload in the scenario. The RUP-baseline and Fair-CO2 are each used to generate their own carbon attribution values for each workload for each scenario. Fairness is evaluated as the deviation in carbon attribution from the ground truth Shapley value attribution.

#### Using Provided Experimental Data
You must have collected and processed colocation experiment data (steps 2a and 2b) to proceed with the Monte Carlo simulations. Alternatively, you may use the provided pre-processed colocation data by copying the contents in <code>colocation/ref-results</code> to <code>colocation/results</code>

### 3a. Dynamic Demand/Workload Schedule Simulations
Run the following script to run the dynamic demand Monte Carlo simulation. 
```
python3 monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py --trials 10000 --max_workloads 22 --min_time_slices 4 --max_time_slices 10 --num_workers 20
```

To run a faster simulation, the simulation scale can be reduced.
```
python3 monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py --trials 1000 --max_workloads 18 --min_time_slices 4 --max_time_slices 8 --num_workers 20
```

Results are stored in <code>monte-carlo-simulations/dynamic-demand/sim-results</code>. Reference results are stored in <code>monte-carlo-simulations/dynamic-demand/ref-sim-results</code>.

Run the following script to generate figure 7 from the paper. The figure will be saved to the <code>figures</code> folder.
```
python3 monte-carlo-simulations/dynamic-demand/gen_dynamic_demand_sim_figures.py
```

### 3b. Colocation Scenario Simulations
Run the following script to run the colocation scenario Monte Carlo simulation.
```
python3 monte-carlo-simulations/colocation/colocation_sim.py  --trials 10000 --min_workloads 4 --max_workloads 100 --min_grid_ci 0 --max_grid_ci 1000 --min_samples 1 --max_samples 15 --num_workers 20
```

To run a faster simulation, the simulation scale can be reduced.
```
python3 monte-carlo-simulations/colocation/colocation_sim.py  --trials 1000 --min_workloads 4 --max_workloads 50 --min_grid_ci 0 --max_grid_ci 500 --min_samples 1 --max_samples 15 --num_workers 20
```

Results are stored in <code>monte-carlo-simulations/colocation/sim-results</code>. Reference results are stored in <code>monte-carlo-simulations/colocation/ref-sim-results</code>.

Run the following script to generate figure 8 and 9 from the paper. The figure will be saved to the <code>figures</code> folder. 
```
python3 monte-carlo-simulations/colocation/gen_colocation_sim_figures.py
```

## 4. Demand Forecasting Evaluation (Figures 5 and 11)
This step evaluates how Fair-CO2 responds to forecasting error. We use [Microsoft Azure 2017 VM traces](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md). We include the 30-day resource allocation time series data from the trace: <code>forecast/azure-time-series.csv</code>. We use [Meta's Prophet tool](https://github.com/facebook/prophet) to forecast the last 9 days of CPU core allocation from the first 21 days of actual data. We apply Fair-CO2 to both the actual full 30-day trace and the generated 30-day trace (w/ 9 days of forecast data and 21 days of actual data) and compare the resultant embodied carbon intensity signals.

Run the following script to perform the demand forecast, generate the embodied carbon signals with Fair-CO2, and generate figures 5 and 11. The figures will be saved to the <code>figures</code> folder.
```
python3 forecast/gen_forecast_eval_figures.py
```

## 5. Workload Carbon Optimization (Figures 10, 12, and 13)
This step runs the runtime parameter sweep experiments for the workload carbon optimization case study. This portion of the experiments takes the longest by far and can run for several days to over a week. Moreover, workloads may crash or stall on low-memory configurations where there is significant memory thrashing as significant portions of the working set resides in swap thus potentially requiring frequent check-ins and intervention.

For this reason, the provided sweep experiment scripts are configured to only do a coarse-grained sweep of CPU and memory allocations. Moreover, we provide reference pre-processed experimental results which can be used to generate the figures in the paper.

#### Set Up A Large Swap Space
Because we are constraining physical memory allocation as part of the experiments in steps 5a and 5b, we will need a large swap space for the workloads to still be able to run. You may also need to delete the FAISS indices if you run out of storage.
```
rm -rf workloads/faiss/indices
source setup-scripts/create_large_swap.sh
```

### 5a. Apache Spark Parameter Sweep
Run this script to sweep across CPU cores and memory for Apache Spark. Results are saved in <code>workload-optimization/results/spark</code>. Note: this can take over a day
```
python3 workload-optimization/spark/spark_sweep.py
```

To run a small subset of the Spark sweep experiments, use the <code>--small</code> argument. Note that this option will not collect enough data to reproduce the results from the paper and should only be used as a demo of the sweep experiments.
```
python3 workload-optimization/spark/spark_sweep.py --small
```

To run a more representative sweep than `--small` but with less experiment points than the full sweep, use the `--medium` option.
```
python3 workload-optimization/spark/spark_sweep.py --medium
```

Run this script to process the raw logs from the Spark sweep. 
```
python3 workload-optimization/spark/process_spark.py
```

#### Using Provided Experimental Data
Alternatively, instead of running the sweep experiments, you may use the included reference pre-processed results in <code>workload-optimization/ref-results/spark</code> by copying its contents to <code>workload-optimization/results/spark</code>.

### 5b. PBBS Parameter Sweep
Run this script to sweep across CPU cores and memory for eight PBBS workloads. Results are saved in <code>workload-optimization/results/pbbs</code>. Note: these experiments will take a very long time to run (over a week)
```
python3 workload-optimization/pbbs/pbbs_sweep.py
```

To run a small subset of the PBBS sweep experiments, use the <code>--small</code> argument. Note that this option will not collect enough data to reproduce the results from the paper and should only be used as a demo of the sweep experiments.
```
python3 workload-optimization/pbbs/pbbs_sweep.py --small
```

To run a more representative sweep than `--small` but with less experiment points than the full sweep, use the `--medium` option.
```
python3 workload-optimization/pbbs/pbbs_sweep.py --medium
```

To run a specific workload from the eight PBBS workloads, specify the desired workload with the `--workload <pbbs_workload>` option where `<pbbs_workload>` can be any one of `{ddup, bfs, msf, wc, sa, ch, nn, nbody}`. This option can be combined with the `--small` option.
```
python3 workload-optimization/pbbs/pbbs_sweep.py --workload <pbbs_workload>
```

Run this script to process the raw logs from the PBBS sweep. 
```
python3 workload-optimization/pbbs/process_pbbs.py
```

#### Using Provided Experimental Data
Alternatively, instead of running the sweep experiments, you may use the included reference pre-processed results in <code>workload-optimization/ref-results/pbbs</code> by copying its contents to <code>workload-optimization/results/pbbs</code>.

### 5c. FAISS Parameter Sweep (Figures 12 and 13)
(Optional) Delete the swap space to free up some storage. We won't need swap space for the FAISS parameter sweep experiment as memory allocation is not a sweep parameter.
```
source setup-scripts/delete_swap.sh
```
(Optional) The IVF and HNSW indices generated will be around 270 GB in size together. If your system is storage space constrained, you may use the following script to remove data and files for other experiments. ONLY DO THIS IF YOU NO LONGER NEED TO RUN ANY PREVIOUS EXPERIMENT.
```
source setup-scripts/purge.sh
```

#### Generate the HNSW FAISS Index (If Not Already Generated)
Note: this is a very lengthy step, taking potentially multiple consecutive days
```
python3 workloads/faiss/gen_HNSW_100M_index.py
```
#### Generate the IVF FAISS Index (If Not Already Generated)
Note: this is a very lengthy step, taking potentially multiple consecutive days
```
python3 workloads/faiss/gen_IVF_SQ8_100M_index.py
```

Run this script to sweep across batch size, CPU cores, and index choice for FAISS. Results are saved in <code>workload-optimization/results/faiss</code>. Note: this can take over a day
```
python3 workload-optimization/faiss/faiss_sweep.py
```

To run a small subset of the FAISS sweep experiments, use the <code>--small</code> argument. Note that this option will not collect enough data to reproduce the results from the paper and should only be used as a demo of the sweep experiments.
```
python3 workload-optimization/faiss/faiss_sweep.py --small
```

To run a more representative sweep than `--small` but with less experiment points than the full sweep, use the `--medium` option.
```
python3 workload-optimization/faiss/faiss_sweep.py --medium
```

Run this script to process the raw logs from the FAISS sweep. 
```
python3 workload-optimization/faiss/process_faiss.py
```

#### Using Provided Experimental Data
Alternatively, instead of running the sweep experiments, you may use the included reference pre-processed results in <code>workload-optimization/ref-results/faiss</code> by copying its contents to <code>workload-optimization/results/faiss</code>.

#### Generating Dynamic Workload Optimization Figures
Run this script to generate figures 12 and 13. The figures will be saved to the <code>figures</code> folder.
```
python3 workload-optimization/gen_dyn_wl_figure.py
```

### 5d. Generate Workload Optimization Summary Figure (Figure 10)
Run these two scripts in order to generate figure 10.
```
python3 workload-optimization/faiss_spark_grid_ci_sweep.py
python3 workload-optimization/gen_sweep_summary_figure.py
```







