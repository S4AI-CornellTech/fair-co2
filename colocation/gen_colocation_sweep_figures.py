import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rel_matrix_filtered(matrix_file, out_file, metric, workloads, with_iso=False):
    matrix = pd.read_csv(matrix_file)
    matrix.set_index('workload', inplace=True)
    if with_iso == False:
        matrix = matrix.drop('nothing', axis=1)
    matrix = matrix.loc[workloads, workloads]
    matrix = matrix.astype(float)
    # Change to percentage
    matrix = matrix * 100
    # Plot the matrix with a heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', fmt='.0f', cbar_kws={'label': f'Change (%)', 'pad': 0.01})
    labels = ['DDUP', 'BFS', 'MSF', 'WC', 'SA', 'CH', 'NN', 'NBODY', 'PG-100', 'PG-50', 'PG-10', 'H.265', 'LLAMA.CPP', 'FAISS', 'SPARK']
    # Annotate with percent sign
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
    # On positive values add a plus sign
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix.iloc[i, j] > 0:
                plt.text(j + 0.5, i + 0.5, f'+{matrix.iloc[i, j]:.0f}%', ha='center', va='center', color='black', fontsize=8)
            else:
                plt.text(j + 0.5, i + 0.5, f'{matrix.iloc[i, j]:.0f}%', ha='center', va='center', color='black', fontsize=8)
    # Y-axis ticks at an angle
    xticks = np.arange(0, len(labels), 1)
    xticks = xticks + 0.5
    yticks = np.arange(0, len(labels), 1)
    yticks = yticks + 0.5
    plt.xticks(rotation=35, ha='right', labels=labels, ticks=xticks, fontsize=12)
    plt.yticks(rotation=0, ha='right', labels=labels, ticks=yticks, fontsize=12)
    plt.ylabel('Workload', fontsize=16)
    # move y-axis label to the left
    plt.gca().yaxis.set_label_coords(-0.12, 0.5)
    # move x-axis label to the left
    plt.xlabel('Colocated Workload', fontsize=16)
    plt.gca().xaxis.set_label_coords(0.45, -0.22)
    # Move the color bar to the right

    # Tweak the margins
    plt.subplots_adjust(left=0.13, right=1.04, top=0.97, bottom=0.22)
    plt.savefig(out_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    fair_co2_path = os.environ.get('FAIR_CO2')
    data_dir = f'{fair_co2_path}/colocation/results'
    figures_dir = f'{fair_co2_path}/figures'
    plot_workload_list = [
        'removeDuplicates',
        'breadthFirstSearch',
        'minSpanningForest',
        'wordCounts',
        'suffixArray',
        'convexHull',
        'nearestNeighbors',
        'nBody',
        'pgbench-100',
        'pgbench-50',
        'pgbench-10',
        'x265',
        'llama',
        'faiss',
        'spark',
    ]   

    runtime_relative_change_matrix_file = f'{data_dir}/runtime_relative_change_matrix.csv'
    plot_rel_matrix_filtered(runtime_relative_change_matrix_file, f'{figures_dir}/2a_runtime_relative_change_matrix_new.png', 'Runtime Change (%)', plot_workload_list, with_iso=False)

    proportional_energy_relative_change_matrix_file = f'{data_dir}/proportional_energy_relative_change_matrix.csv'
    plot_rel_matrix_filtered(proportional_energy_relative_change_matrix_file, f'{figures_dir}/2b_proportional_energy_relative_change_matrix_new.png', 'Energy Change (%)', plot_workload_list, with_iso=False)