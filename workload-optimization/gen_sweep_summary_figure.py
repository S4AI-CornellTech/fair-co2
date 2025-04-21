import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

fair_co2_path = os.environ.get('FAIR_CO2')
figures_dir = f'{fair_co2_path}/figures'
data_dir = f'{fair_co2_path}/workload-optimization/results'

# Hardcoded label box positions
wc_label_pos = [(2.5, 0.88), (1.5, 1.09), (0.5, 0.4), (0.5, 0.5), (0.5, 0.5)]
sa_label_pos = [(1.3, 0.85), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6)]
ddup_label_pos = [(0.75, 0.7), (0.5, 0.8), (0.5, 0.9), (0.5, 0.9), (0.5, 0.9)]
nn_label_pos = [(0.5, 1.05), (0.5, 0.8), (0.5, 0.9), (0.5, 0.9), (0.5, 0.9)]
faiss_label_pos = [(2.9, 0.915), (0.7, 0.74), (0.5, 0.85), (0.5, 0.7), (0.5, 0.7)]
nbody_label_pos = [(0.6, 0.7), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8)]
msf_label_pos = [(2.6, 1.2), (1.1, 0.73), (1, 0.88), (0.5, 0.8), (0.5, 1.0)]
ch_label_pos = [(0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8)]
bfs_label_pos = [(0.95, 0.6), (1.5, 0.75), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8)]
spark_label_pos = [(2, 2.3), (1.2, 0.6), (1.5, 1.4), (0.6, 1.4), (0.5, 1.4)]

label_pos = {
    'wordCounts': wc_label_pos,
    'suffixArray': sa_label_pos,
    'removeDuplicates': ddup_label_pos,
    'nearestNeighbors': nn_label_pos,
    'faiss': faiss_label_pos,
    'nBody': nbody_label_pos,
    'minSpanningForest': msf_label_pos,
    'convexHull': ch_label_pos,
    'breadthFirstSearch': bfs_label_pos,
    'spark': spark_label_pos
}

def plot_summary_normalized_2x5(workload_list, workload_label_list, grid_ci_list):
    fig, axs = plt.subplots(2, 5, figsize=(20, 5.5), sharex=True, sharey=False)
    for i, workload in enumerate(workload_list):
        if workload == 'spark':
            # Read from pickle
            with open(f'{data_dir}/spark/spark_grid_ci_sweep.pkl', 'rb') as f:
                workload_df = pickle.load(f)
            # workload_df = pd.read_csv(f'{data_dir}/spark/spark_grid_ci_sweep.csv')
        elif workload == 'faiss':
            # Read from pickle
            with open(f'{data_dir}/faiss/faiss_grid_ci_sweep.pkl', 'rb') as f:
                workload_df = pickle.load(f)
            # workload_df = pd.read_csv(f'{data_dir}/faiss/faiss_grid_ci_sweep.csv')
        else:
            workload_df = pd.read_csv(f'{data_dir}/pbbs/{workload}/filtered_co2_pbbs_data.csv')
        # Plot grid CI on the x-axis and total carbon footprint on the y-axis
        # Sort by grid CI
        workload_df = workload_df.sort_values(by='Grid CI (gCO2eq/kWh)')
        if workload == 'spark' or workload == 'faiss':
            min_cf = workload_df['Min CF CF (gCO2eq)']
            min_runtime = workload_df['Min Runtime CF (gCO2eq)']
            min_energy = workload_df['Min Energy CF (gCO2eq)']
            min_embodied = workload_df['Min Embodied CF (gCO2eq)']
            min_cf_config = workload_df['Min CF Config']
        else:
            min_cf = workload_df[workload_df['Min CF'] == True]['Total Carbon Footprint'].values
            min_runtime = workload_df[workload_df['Min Runtime'] == True]['Total Carbon Footprint'].values
            min_energy = workload_df[workload_df['Min Energy'] == True]['Total Carbon Footprint'].values
            min_embodied = workload_df[workload_df['Min Embodied'] == True]['Total Carbon Footprint'].values
        # Normalize the carbon footprints elementwise
        min_cf = min_cf / min_runtime
        min_energy = min_energy / min_runtime
        min_embodied = min_embodied / min_runtime
        min_runtime = min_runtime / min_runtime

        # Shade each region when the min CF configuration changes
        regions = []
        region_batch_size = []
        region_index = []
        region_cpu = []
        region_memory = []
        region_colors = []
        
        if workload == 'spark':
            min_cf_cpu = (min_cf_config.values[0])[1]
            min_cf_memory = (min_cf_config.values[0])[0]
        elif workload == 'faiss':
            min_cf_index = (min_cf_config.values[0])[0]
            min_cf_cpu = (min_cf_config.values[0])[1]
            min_cf_batch_size = (min_cf_config.values[0])[2]
        else:
            min_cf_rows = workload_df[workload_df['Min CF'] == True]
            min_cf_cpu = min_cf_rows[min_cf_rows['Grid CI (gCO2eq/kWh)'] == grid_ci_list[0]]['cpus'].values[0]
            min_cf_memory = min_cf_rows[min_cf_rows['Grid CI (gCO2eq/kWh)'] == grid_ci_list[0]]['memory'].values[0]
        region_start_cf = grid_ci_list[0]

        if workload == 'spark':
            for j, grid_ci in enumerate(grid_ci_list):
                if j == 0:
                    continue
                min_cf_cpu_next = (min_cf_config.values[j])[1]
                min_cf_memory_next = (min_cf_config.values[j])[0]
                if min_cf_cpu_next != min_cf_cpu or min_cf_memory_next != min_cf_memory:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_cpu.append(min_cf_cpu)
                    region_memory.append(min_cf_memory)
                    region_start_cf = grid_ci
                    min_cf_cpu = min_cf_cpu_next
                    min_cf_memory = min_cf_memory_next 
                if grid_ci == grid_ci_list[-1]:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_cpu.append(min_cf_cpu)
                    region_memory.append(min_cf_memory)
        elif workload == 'faiss':
            for j, grid_ci in enumerate(grid_ci_list):
                if j == 0:
                    continue
                min_cf_index_next = (min_cf_config.values[j])[0]
                min_cf_cpu_next = (min_cf_config.values[j])[1]
                min_cf_batch_size_next = (min_cf_config.values[j])[2]
                if min_cf_cpu_next != min_cf_cpu or min_cf_memory_next != min_cf_memory or min_cf_index_next != min_cf_index:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_index.append(min_cf_index)
                    region_cpu.append(min_cf_cpu)
                    region_batch_size.append(min_cf_batch_size)
                    region_start_cf = grid_ci
                    min_cf_index = min_cf_index_next
                    min_cf_cpu = min_cf_cpu_next
                    min_cf_batch_size = min_cf_batch_size_next
                if grid_ci == grid_ci_list[-1]:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_index.append(min_cf_index)
                    region_cpu.append(min_cf_cpu)
                    region_batch_size.append(min_cf_batch_size)
        else:
            for grid_ci in grid_ci_list[1:]:
                min_cf_cpu_next = min_cf_rows[min_cf_rows['Grid CI (gCO2eq/kWh)'] == grid_ci]['cpus'].values[0]
                min_cf_memory_next = min_cf_rows[min_cf_rows['Grid CI (gCO2eq/kWh)'] == grid_ci]['memory'].values[0]
                if min_cf_cpu_next != min_cf_cpu or min_cf_memory_next != min_cf_memory:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_cpu.append(min_cf_cpu)
                    region_memory.append(min_cf_memory)
                    region_start_cf = grid_ci
                    min_cf_cpu = min_cf_cpu_next
                    min_cf_memory = min_cf_memory_next 
                if grid_ci == grid_ci_list[-1]:
                    region_end_cf = grid_ci
                    regions.append((region_start_cf, region_end_cf))
                    region_cpu.append(min_cf_cpu)
                    region_memory.append(min_cf_memory)


        linewidth = 4
        axs[i//5, i%5].plot(grid_ci_list, min_runtime[:len(grid_ci_list)], color='red', linestyle='solid', linewidth=linewidth)
        axs[i//5, i%5].plot(grid_ci_list, min_energy[:len(grid_ci_list)], color='blue', linestyle='dotted', linewidth=linewidth)
        axs[i//5, i%5].plot(grid_ci_list, min_embodied[:len(grid_ci_list)], color='orange', linestyle='dashdot', linewidth=linewidth)
        axs[i//5, i%5].plot(grid_ci_list, min_cf[:len(grid_ci_list)], color='green', linestyle='dashed', linewidth=linewidth)
        fig.legend(['Min. Runtime', 'Min. Operational', 'Min. Embodied', 'Min. Carbon Footprint'], loc='upper center', ncol=4, fontsize=18)
        region_colors = ['green', 'lightgreen']
        region_offsets = [0, 0.2, -0.1, 0.1, -0.2]
        for k, region in enumerate(regions):
            axs[i//5, i%5].axvspan(region[0], region[1], color=region_colors[k % 2], alpha=0.3)
            bbox_props = dict(boxstyle="round,pad=0.3", fc=region_colors[k % 2], ec="black", lw=2, alpha=0.5)
            (ylim_min, ylim_max) = axs[i//5, i%5].get_ylim()
            # Offset boxes so they don't overlap
            # if workload == 'faiss':
            #     axs[i//5, i%5].text((region[0] + region[1])/2, ylim_min + (ylim_max - ylim_min)*(0.6 + region_offsets[k%5]), f'{region_index[k]} \n{region_cpu[k]} c\n{region_batch_size[k]} batch', ha="center", va="center", rotation=0, bbox=bbox_props, fontsize=9)
            # else:
            #     axs[i//5, i%5].text((region[0] + region[1])/2, ylim_min + (ylim_max - ylim_min)*(0.6 + region_offsets[k%5]), f'{region_cpu[k]} c\n{region_memory[k]} GB', ha="center", va="center", rotation=0, bbox=bbox_props, fontsize=9)

            if workload == 'faiss':
                if region_index[k] == 'ivf': index_label = 'IVF'
                elif region_index[k] == 'hnsw': index_label = 'HNSW'

                axs[i//5, i%5].text(region[0] + (region[1] - region[0])*label_pos[workload][k][0], label_pos[workload][k][1], f'{index_label} \n{int(region_cpu[k])} C\n{int(region_batch_size[k])} BS', ha="center", va="center", rotation=0, bbox=bbox_props, fontsize=10)
            else:
                axs[i//5, i%5].text(region[0] + (region[1] - region[0])*label_pos[workload][k][0], label_pos[workload][k][1], f'{int(region_cpu[k])} C\n{int(region_memory[k])} GB', ha="center", va="center", rotation=0, bbox=bbox_props, fontsize=10)
        axs[i//5, i%5].set_title(workload_label_list[i], fontsize = 20)
        axs[i//5, i%5].set_xlim(0, 400)
        # Set x-axis ticks
        axs[i//5, i%5].set_xticks(range(0, 410, 100))
        axs[i//5, 0].set_ylabel(' ')
        axs[1, i%5].set_xlabel(' ')
        axs[i//5, i%5].tick_params(axis='y', which='major', labelsize=16)
        axs[i//5, i%5].tick_params(axis='x', which='major', labelsize=16)

    # Label the x-axis centered in the whole plot
    fig.text(0.5, 0.02, 'Grid Carbon Intensity ($gCO_2eq/kWh$)', ha='center', fontsize=20)
    # Label the y-axis centered in the whole plot
    fig.text(0, 0.5, 'Norm. Carbon Footprint', va='center', rotation='vertical', fontsize=20)
    
    fig.tight_layout()
    # Add more space at the top for the legend
    fig.subplots_adjust(bottom=0.13, top=0.83)
    plt.savefig(f'{figures_dir}/10_summary_normalized_2x5.pdf')
    plt.close()


if __name__ == '__main__':
    workload_list_2x5 = [
        'wordCounts',
        'suffixArray',
        'removeDuplicates',
        'nearestNeighbors',
        'faiss',
        'nBody',
        'minSpanningForest',
        'convexHull',
        'breadthFirstSearch',
        'spark',
    ]   
    workload_label_list = [
        'WC',
        'SA',
        'DDUP',
        'NN',
        'FAISS',
        'NBODY',
        'MSF',
        'CH',
        'BFS',
        'SPARK',
    ]
    pbbs_workload_list = [
        'wordCounts',
        'suffixArray',
        'removeDuplicates',
        'nearestNeighbors',
        'nBody',
        'minSpanningForest',
        'convexHull',
        'breadthFirstSearch',
    ]   
    grid_ci_list = range (0, 410, 10)
    plot_summary_normalized_2x5(workload_list_2x5, workload_label_list, grid_ci_list)