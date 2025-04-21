import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import os
# plt.rcParams['text.usetex'] = True

fair_co2_path = os.environ.get('FAIR_CO2')

# Set global style
figsize = (6, 4)
xlabel_fontsize = 22
ylabel_fontsize = 22
tick_fontsize = 20
title_fontsize = 20
legend_fontsize = 17.5
dpi = 200
base_palette = "Dark2"
palette = sns.color_palette(base_palette)
# Swap the second and third colors
palette[1], palette[2] = palette[2], palette[1]
palette_1 = palette[1:]
palette_2 = palette[2:]
scatter_marker_size = 20
scatter_alpha = 0.3
legend_markersize = 20
legend_alpha = 1
edgewidth = 0
linewidth = 1
violin_width = 0.8
baseline_name = 'RUP-Baseline'
demand_proportional_name = 'Demand-Prop.'
temporal_shapley_name = 'FAIR-CO2'

# Fig box and whisker comparison avg and max deviation from shapley attributions for different methods
def violinplot_all_values(df_val, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = df_val[df_val['Attribution Method'] == temporal_shapley_name]
    vp = sns.violinplot(
        data=df_filtered, x="Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=.6, palette=palette, ax=ax, linewidth=linewidth, 
    )
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/all_values_violin_absolute_time_slices.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_all_values_absolute(df_val, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df = df_val.copy()
    df['Deviation (%)'] = np.abs(df['Deviation (%)'])
    vp = sns.violinplot(
        data=df, x="Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=.6, palette=palette, ax=ax, linewidth=linewidth,
    )
    vp.set(ylabel=None)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/all_values_violin_absolute.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_all_values_absolute_time_slices(df_val, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df = df_val.copy()
    df['Deviation (%)'] = np.abs(df['Deviation (%)'])
    df_filtered = df[df['Attribution Method'] == temporal_shapley_name]
    vp = sns.violinplot(
        data=df_filtered, x="Num Time Slices", y="Deviation (%)", hue="Attribution Method",
        width=.6, palette=palette, ax=ax, linewidth=linewidth
    )
    vp.set(ylabel=None)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/all_values_violin_absolute_time_slices.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def box_and_whisker_overall_avg(df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    bp = sns.boxplot(
        data=df, x="Average Deviation (%)", y="Attribution Method", hue="Attribution Method",
        whis=[0, 100], width=.6, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    bp.set(ylabel=None)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/overall_deviation_box_and_whisker_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_overall_avg(df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    vp = sns.violinplot(
        data=df, x="Average Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=0.6, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    vp.set(ylabel=None)
    # Remove y-axis ticks
    vp.set(yticklabels=[])
    vp.tick_params(left=False)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.xlim(-10,220)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Average Deviation', fontsize=title_fontsize)
    # Add legend with color and labels for each attribution method
    plt.legend(bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes, \
               fontsize=legend_fontsize)
    # plt.legend(title='Method', loc='lower right', labels=['Baseline', 'Interference-Aware'], fontsize=legend_fontsize)   
    plt.tight_layout()
    output_file = f'{output_dir}/7a_overall_deviation_violin_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return


# Fig box and whisker comparison max deviation from shapley attributions for different methods
def box_and_whisker_overall_worst(df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    bp = sns.boxplot(
        data=df, x="Worst-Case Deviation (%)", y="Attribution Method", hue="Attribution Method",
        whis=[0, 100], width=.6, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    bp.set(ylabel=None)
    # Set legend fontsize

    plt.xlabel('Deviation from Ground Truth (%)', fontsize=xlabel_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    # plt.title(f' Worst Case Deviation (%)', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/overall_deviation_box_and_whisker_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_overall_worst(df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    vp = sns.violinplot(
        data=df, x="Worst-Case Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=0.6, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    vp.set(ylabel=None)
    vp.set(yticklabels=[])
    vp.tick_params(left=False)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.xlim(-50,800)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Worst-Case Deviation', fontsize=title_fontsize)
    plt.legend(bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes, \
            fontsize=legend_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/7e_overall_deviation_violin_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

# fig box and whisker for different time_slices avg and max deviation from shapley attributions
def box_and_whisker_time_slices_avg(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = synthetic_schedule_data_df[synthetic_schedule_data_df['Attribution Method'] == temporal_shapley_name]
    bp = sns.boxplot(
        data=df_filtered, x="Num Time Slices", y="Average Deviation (%)", hue="Attribution Method",
        whis=[0, 100], width=.6, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Samples', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/n_time_slices_deviation_box_and_whisker_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_time_slices_avg(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    vp = sns.violinplot(
        data=synthetic_schedule_data_df, x="Num Time Slices", y="Average Deviation (%)", hue="Attribution Method",
        width=.9, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.ylim(-10, 170)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/7b_n_time_slices_deviation_violin_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_time_slices_avg_only_demand(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    df = synthetic_schedule_data_df[synthetic_schedule_data_df['Attribution Method'] != baseline_name]
    vp = sns.violinplot(
        data=df, x="Num Time Slices", y="Average Deviation (%)", hue="Attribution Method",
        width=.8, palette=palette_1, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.ylim(-5, 70)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/7c_n_time_slices_deviation_violin_avg_only_demand.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_time_slices_avg_only_temporal_shapley(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    df = synthetic_schedule_data_df[synthetic_schedule_data_df['Attribution Method'] == temporal_shapley_name]
    vp = sns.violinplot(
        data=df, x="Num Time Slices", y="Average Deviation (%)", hue="Attribution Method",
        width=.6, palette=palette_2, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.ylim(-10, 40)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/n_time_slices_deviation_violin_avg_only_temporal_shapley.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def box_and_whisker_time_slices_worst(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = synthetic_schedule_data_df[synthetic_schedule_data_df['Attribution Method'] == temporal_shapley_name]
    sns.boxplot(
        data=df_filtered, x="Num Time Slices", y="Worst-Case Deviation (%)",
        whis=[0, 100], width=violin_width, ax=ax, legend=False, linewidth=linewidth
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/n_time_slices_deviation_box_and_whisker_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_time_slices_worst(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    vp = sns.violinplot(
        data=synthetic_schedule_data_df, x="Num Time Slices", y="Worst-Case Deviation (%)", hue="Attribution Method",
        width=.9, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.ylim(-50, 550)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/7f_time_slices_deviation_violin_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_time_slices_worst_only_demand(synthetic_schedule_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    df = synthetic_schedule_data_df[synthetic_schedule_data_df['Attribution Method'] != baseline_name]
    vp = sns.violinplot(
        data=df, x="Num Time Slices", y="Worst-Case Deviation (%)", hue="Attribution Method",
        width=.8, palette=palette_1, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Time Slices', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.ylim(-20, 250)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/7g_n_time_slices_deviation_violin_worst_only_demand.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

# Avg dev scatter over num workloads
def scatter_num_workloads_avg(df, p95_df, avg_df, output_dir):
    plt.figure(figsize=figsize)
    # Plot one scatter for each attribution method on the same plot
    sp = sns.scatterplot(
        data=df, x="Num Workloads", y="Average Deviation (%)", hue="Attribution Method",
        palette=palette, s=scatter_marker_size, alpha=scatter_alpha, linewidth=edgewidth
    )
    handles, labels = sp.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(legend_markersize)
        h.set_alpha(legend_alpha)

    # Plot the 95th percentile of deviation for each method
    p95_baseline_df = p95_df[p95_df['Attribution Method'] == baseline_name]
    p95_demand_proportional_df = p95_df[p95_df['Attribution Method'] == demand_proportional_name]
    p95_temporal_shapley_df = p95_df[p95_df['Attribution Method'] == temporal_shapley_name]
    sp.plot(p95_baseline_df['Num Workloads'], p95_baseline_df['95th Percentile Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_demand_proportional_df['Num Workloads'], p95_demand_proportional_df['95th Percentile Average Deviation (%)'],
            color='indigo', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_temporal_shapley_df['Num Workloads'], p95_temporal_shapley_df['95th Percentile Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_demand_proportional_df = avg_df[avg_df['Attribution Method'] == demand_proportional_name]
    avg_temporal_shapley_df = avg_df[avg_df['Attribution Method'] == temporal_shapley_name]

    sp.plot(avg_baseline_df['Num Workloads'], avg_baseline_df['Average Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_demand_proportional_df['Num Workloads'], avg_demand_proportional_df['Average Average Deviation (%)'],
            color='indigo', linewidth=linewidth*1.5)
    sp.plot(avg_temporal_shapley_df['Num Workloads'], avg_temporal_shapley_df['Average Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)

    plt.ylim(0, 270)

    plt.xlabel('Number of Workloads', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # Plot x ticks as integers
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    output_file = f'{output_dir}/7d_num_workloads_deviation_scatter_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def scatter_num_workloads_worst(df, p95_df, avg_df, output_dir):
    plt.figure(figsize=figsize)
    # Plot one scatter for each attribution method on the same plot
    sp = sns.scatterplot(
        data=df, x="Num Workloads", y="Worst-Case Deviation (%)", hue="Attribution Method",
        palette=palette, s=scatter_marker_size, alpha=scatter_alpha, linewidth=edgewidth
    )
    handles, labels = sp.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(legend_markersize)
        h.set_alpha(legend_alpha)

    # Plot the 95th percentile of deviation for each method
    p95_baseline_df = p95_df[p95_df['Attribution Method'] == baseline_name]
    p95_demand_proportional_df = p95_df[p95_df['Attribution Method'] == demand_proportional_name]
    p95_temporal_shapley_df = p95_df[p95_df['Attribution Method'] == temporal_shapley_name]
    sp.plot(p95_baseline_df['Num Workloads'], p95_baseline_df['95th Percentile Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_demand_proportional_df['Num Workloads'], p95_demand_proportional_df['95th Percentile Worst-Case Deviation (%)'],
            color='indigo', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_temporal_shapley_df['Num Workloads'], p95_temporal_shapley_df['95th Percentile Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_demand_proportional_df = avg_df[avg_df['Attribution Method'] == demand_proportional_name]
    avg_temporal_shapley_df = avg_df[avg_df['Attribution Method'] == temporal_shapley_name]

    sp.plot(avg_baseline_df['Num Workloads'], avg_baseline_df['Average Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_demand_proportional_df['Num Workloads'], avg_demand_proportional_df['Average Worst-Case Deviation (%)'],
            color='indigo', linewidth=linewidth*1.5)
    sp.plot(avg_temporal_shapley_df['Num Workloads'], avg_temporal_shapley_df['Average Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)

    plt.ylim(0, 770)

    plt.xlabel('Number of Workloads', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # Plot x ticks as integers
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    output_file = f'{output_dir}/7h_num_workloads_deviation_scatter_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def process_synthetic_schedule_data_chunk(synthetic_schedule_data_df_chunk):
    df_chunk = pd.DataFrame(columns=['Attribution Method', 'Average Deviation (%)', 'Worst-Case Deviation (%)', 'Num Workloads', 'Num Time Slices'])
    num_simulations = len(synthetic_schedule_data_df_chunk)
    # Process
    for i, row in synthetic_schedule_data_df_chunk.iterrows():
        df_chunk = df_chunk._append({'Attribution Method': baseline_name, 
                        'Average Deviation (%)': row['baseline_avg_deviation (%)'], 
                        'Worst-Case Deviation (%)': row['baseline_worst_case_deviation (%)'],
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': demand_proportional_name,
                        'Average Deviation (%)': row['demand_proportional_avg_deviation (%)'],
                        'Worst-Case Deviation (%)': row['demand_proportional_worst_case_deviation (%)'],
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': temporal_shapley_name, 
                        'Average Deviation (%)': row['temporal_shapley_avg_deviation (%)'], 
                        'Worst-Case Deviation (%)': row['temporal_shapley_worst_case_deviation (%)'],
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices']}, ignore_index=True)

    return df_chunk

def process_synthetic_schedule_data(synthetic_schedule_data_df_file):
    synthetic_schedule_data_df = pd.read_csv(synthetic_schedule_data_df_file)
    synthetic_schedule_data_df = synthetic_schedule_data_df.astype({'num_workloads': 'int32', 
                                                        'num_time_slices': 'int32', 
                                                        'baseline_avg_deviation (%)': 'float64',
                                                        'demand_proportional_avg_deviation (%)': 'float64',
                                                        'temporal_shapley_avg_deviation (%)': 'float64', 
                                                        'baseline_worst_case_deviation (%)': 'float64',
                                                        'demand_proportional_worst_case_deviation (%)': 'float64',
                                                        'temporal_shapley_worst_case_deviation (%)': 'float64'})
    df = pd.DataFrame(columns=['Attribution Method', 'Average Deviation (%)', 'Worst-Case Deviation (%)', 'Num Workloads', 'Num Time Slices'])
    num_simulations = len(synthetic_schedule_data_df)
    # Process in chunks in parallel
    num_chunks = 10
    chunk_size = num_simulations // num_chunks
    synthetic_schedule_data_df_chunks = [synthetic_schedule_data_df[i*chunk_size:(i+1)*chunk_size] for i in range(0, num_chunks)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        df_chunks = executor.map(process_synthetic_schedule_data_chunk, synthetic_schedule_data_df_chunks)
    df = pd.concat(df_chunks)
    df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_processed.csv')

def process_synthetic_schedule_val_data_chunk(synthetic_schedule_val_data_df_chunk):
    df_chunk = pd.DataFrame(columns=['Attribution Method', 'Deviation (%)', 'Num Workloads', 'Num Time Slices', 'CPUs', 'Runtime'])
    num_simulations = len(synthetic_schedule_val_data_df_chunk)
    for i, row in synthetic_schedule_val_data_df_chunk.iterrows():
        df_chunk = df_chunk._append({'Attribution Method': baseline_name, 
                        'Deviation (%)': row['baseline_deviation (%)'], 
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices'],
                        'CPUs': row['workload_cpus'],
                        'Runtime': row['workload_runtime']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': demand_proportional_name,
                        'Deviation (%)': row['demand_proportional_deviation (%)'],
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices'],
                        'CPUs': row['workload_cpus'],
                        'Runtime': row['workload_runtime']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': temporal_shapley_name, 
                        'Deviation (%)': row['temporal_shapley_deviation (%)'], 
                        'Num Workloads': row['num_workloads'],
                        'Num Time Slices': row['num_time_slices'],
                        'CPUs': row['workload_cpus'],
                        'Runtime': row['workload_runtime']}, ignore_index=True)

    return df_chunk

def process_synthetic_schedule_val_data(synthetic_schedule_val_data_df_file):
    synthetic_schedule_data_df = pd.read_csv(synthetic_schedule_val_data_df_file)
    synthetic_schedule_data_df = synthetic_schedule_data_df.astype({'num_workloads': 'int32', 
                                                        'num_time_slices': 'int32', 
                                                        'workload_cpus': 'int32',
                                                        'workload_runtime': 'float64',
                                                        'baseline_deviation (%)': 'float64', 
                                                        'demand_proportional_deviation (%)': 'float64',
                                                        'temporal_shapley_deviation (%)': 'float64'})
    df = pd.DataFrame(columns=['Attribution Method', 'Deviation (%)', 'Num Workloads', 'Num Time Slices', 'CPUs', 'Runtime'])
    num_chunks = 10
    chunk_size = len(synthetic_schedule_data_df) // num_chunks
    synthetic_schedule_data_df_chunks = [synthetic_schedule_data_df[i*chunk_size:(i+1)*chunk_size] for i in range(0, num_chunks)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        df_chunks = executor.map(process_synthetic_schedule_val_data_chunk, synthetic_schedule_data_df_chunks)
    df = pd.concat(df_chunks)
    df = df.astype({'Num Workloads': 'int32', 'Num Time Slices': 'int32', 'CPUs': 'int32'})
    df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_val_processed.csv')

def process_p95_num_workloads_data(df):
    # For each num workloads, find the 95th percentile of deviation for each attribution method
    max_num_workloads = df['Num Workloads'].max()
    min_num_workloads = df['Num Workloads'].min()
    num_workloads = np.arange(min_num_workloads, max_num_workloads + 1, 1)
    p95_df = pd.DataFrame(columns=['Attribution Method', 'Num Workloads', '95th Percentile Average Deviation (%)', '95th Percentile Worst-Case Deviation (%)'])
    p95_avg_baseline_dev_list = []
    p95_avg_demand_proportional_dev_list = []
    p95_avg_temporal_shapley_dev_list = []
    p95_worst_baseline_dev_list = []
    p95_worst_demand_proportional_dev_list = []
    p95_worst_temporal_shapley_dev_list = []
    for i in range(len(num_workloads)):
        num_workload = num_workloads[i]
        df_filtered = df[df['Num Workloads'] == num_workload]
        print(f'Num workloads: {num_workload}, num time_slices: {len(df_filtered)}')
        p95_avg_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'], 95)
        p95_avg_demand_proportional_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == demand_proportional_name]['Average Deviation (%)'], 95)
        p95_avg_temporal_shapley_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == temporal_shapley_name]['Average Deviation (%)'], 95)
        p95_worst_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'], 95)
        p95_worst_demand_proportional_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == demand_proportional_name]['Worst-Case Deviation (%)'], 95)
        p95_worst_temporal_shapley_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == temporal_shapley_name]['Worst-Case Deviation (%)'], 95)
        p95_avg_baseline_dev_list.append(p95_avg_baseline_dev)
        p95_avg_demand_proportional_dev_list.append(p95_avg_demand_proportional_dev)
        p95_avg_temporal_shapley_dev_list.append(p95_avg_temporal_shapley_dev)
        p95_worst_baseline_dev_list.append(p95_worst_baseline_dev)
        p95_worst_demand_proportional_dev_list.append(p95_worst_demand_proportional_dev)
        p95_worst_temporal_shapley_dev_list.append(p95_worst_temporal_shapley_dev)
    
    # Apply moving average to smooth the curve
    sigma = 2
    p95_avg_baseline_dev_list = gaussian_filter1d(p95_avg_baseline_dev_list, sigma=sigma)
    p95_avg_demand_proportional_dev_list = gaussian_filter1d(p95_avg_demand_proportional_dev_list, sigma=sigma)
    p95_avg_temporal_shapley_dev_list = gaussian_filter1d(p95_avg_temporal_shapley_dev_list, sigma=sigma)
    p95_worst_baseline_dev_list = gaussian_filter1d(p95_worst_baseline_dev_list, sigma=sigma)
    p95_worst_demand_proportional_dev_list = gaussian_filter1d(p95_worst_demand_proportional_dev_list, sigma=sigma)
    p95_worst_temporal_shapley_dev_list = gaussian_filter1d(p95_worst_temporal_shapley_dev_list, sigma=sigma)

    # Add to p95_df
    for i in range(len(p95_avg_baseline_dev_list)):
        p95_df = p95_df._append({'Attribution Method': baseline_name,
                                 'Num Workloads': num_workloads[i], 
                                '95th Percentile Average Deviation (%)': p95_avg_baseline_dev_list[i],
                                '95th Percentile Worst-Case Deviation (%)': p95_worst_baseline_dev_list[i]}, ignore_index=True)
        p95_df = p95_df._append({'Attribution Method': demand_proportional_name,
                                    'Num Workloads': num_workloads[i], 
                                    '95th Percentile Average Deviation (%)': p95_avg_demand_proportional_dev_list[i],
                                    '95th Percentile Worst-Case Deviation (%)': p95_worst_demand_proportional_dev_list[i]}, ignore_index=True)
        p95_df = p95_df._append({'Attribution Method': temporal_shapley_name,
                                    'Num Workloads': num_workloads[i], 
                                    '95th Percentile Average Deviation (%)': p95_avg_temporal_shapley_dev_list[i],
                                    '95th Percentile Worst-Case Deviation (%)': p95_worst_temporal_shapley_dev_list[i]}, ignore_index=True)
    p95_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_p95_num_workloads.csv')
    return p95_df

def process_avg_num_workloads_data(df):
    # For each num workloads, find the average deviation for each attribution method
    max_num_workloads = df['Num Workloads'].max()
    min_num_workloads = df['Num Workloads'].min()
    num_workloads = np.arange(min_num_workloads, max_num_workloads + 1, 1)
    avg_df = pd.DataFrame(columns=['Attribution Method', 'Num Workloads', 'Average Average Deviation (%)', 'Average Worst-Case Deviation (%)'])
    avg_avg_baseline_dev_list = []
    avg_avg_demand_proportional_dev_list = []
    avg_avg_temporal_shapley_dev_list = []
    avg_worst_baseline_dev_list = []
    avg_worst_demand_proportional_dev_list = []
    avg_worst_temporal_shapley_dev_list = []
    for i in range(len(num_workloads)):
        num_workload = num_workloads[i]
        df_filtered = df[df['Num Workloads'] == num_workload]
        avg_avg_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'])
        avg_avg_demand_proportional_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == demand_proportional_name]['Average Deviation (%)'])
        avg_avg_temporal_shapley_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == temporal_shapley_name]['Average Deviation (%)'])
        avg_worst_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'])
        avg_worst_demand_proportional_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == demand_proportional_name]['Worst-Case Deviation (%)'])
        avg_worst_temporal_shapley_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == temporal_shapley_name]['Worst-Case Deviation (%)'])
        avg_avg_baseline_dev_list.append(avg_avg_baseline_dev)
        avg_avg_demand_proportional_dev_list.append(avg_avg_demand_proportional_dev)
        avg_avg_temporal_shapley_dev_list.append(avg_avg_temporal_shapley_dev)
        avg_worst_baseline_dev_list.append(avg_worst_baseline_dev)
        avg_worst_demand_proportional_dev_list.append(avg_worst_demand_proportional_dev)
        avg_worst_temporal_shapley_dev_list.append(avg_worst_temporal_shapley_dev)

    # Apply moving average to smooth the curve
    sigma = 2
    avg_avg_baseline_dev_list = gaussian_filter1d(avg_avg_baseline_dev_list, sigma=sigma)
    avg_avg_demand_proportional_dev_list = gaussian_filter1d(avg_avg_demand_proportional_dev_list, sigma=sigma)
    avg_avg_temporal_shapley_dev_list = gaussian_filter1d(avg_avg_temporal_shapley_dev_list, sigma=sigma)
    avg_worst_baseline_dev_list = gaussian_filter1d(avg_worst_baseline_dev_list, sigma=sigma)
    avg_worst_demand_proportional_dev_list = gaussian_filter1d(avg_worst_demand_proportional_dev_list, sigma=sigma)
    avg_worst_temporal_shapley_dev_list = gaussian_filter1d(avg_worst_temporal_shapley_dev_list, sigma=sigma)

    # Add to avg_df
    for i in range(len(avg_avg_baseline_dev_list)):
        avg_df = avg_df._append({'Attribution Method': baseline_name,
                                 'Num Workloads': num_workloads[i], 
                                'Average Average Deviation (%)': avg_avg_baseline_dev_list[i],
                                'Average Worst-Case Deviation (%)': avg_worst_baseline_dev_list[i]}, ignore_index=True)
        avg_df = avg_df._append({'Attribution Method': demand_proportional_name,
                                    'Num Workloads': num_workloads[i], 
                                    'Average Average Deviation (%)': avg_avg_demand_proportional_dev_list[i],
                                    'Average Worst-Case Deviation (%)': avg_worst_demand_proportional_dev_list[i]}, ignore_index=True)
        avg_df = avg_df._append({'Attribution Method': temporal_shapley_name,
                                    'Num Workloads': num_workloads[i], 
                                    'Average Average Deviation (%)': avg_avg_temporal_shapley_dev_list[i],
                                    'Average Worst-Case Deviation (%)': avg_worst_temporal_shapley_dev_list[i]}, ignore_index=True)
    avg_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_avg_num_workloads.csv')
    return avg_df


def main():
    process_synthetic_schedule_data(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule.csv')
    process_synthetic_schedule_val_data(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_vals.csv')
    df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_processed.csv')
    df = df.astype({'Num Workloads': 'int32', 
                    'Num Time Slices': 'int32', 
                    'Average Deviation (%)': 'float64', 
                    'Worst-Case Deviation (%)': 'float64'})
    df_val = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_val_processed.csv')
    df_val = df_val.astype({'Num Workloads': 'int32', 
                            'Num Time Slices': 'int32',
                            'Deviation (%)': 'float64',
                            'CPUs': 'int32',
                            'Runtime': 'float64'})
    # Rename Demand-Proportional to Demand-Prop.
    df['Attribution Method'] = df['Attribution Method'].replace({'Demand-Proportional': 'Demand-Prop.'})
    df_val['Attribution Method'] = df_val['Attribution Method'].replace({'Demand-Proportional': 'Demand-Prop.'})
    # Process data
    p95_num_workloads_df = process_p95_num_workloads_data(df)
    avg_num_workloads_df = process_avg_num_workloads_data(df)
    # Save p95_num_workloads_df and avg_num_workloads_df to csv
    p95_num_workloads_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_p95_num_workloads.csv')
    avg_num_workloads_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_avg_num_workloads.csv')
    
    # # Read p95_num_workloads_df and avg_num_workloads_df from csv
    # p95_num_workloads_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_p95_num_workloads.csv')
    # avg_num_workloads_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/dynamic-demand/sim-results/synthetic_schedule_avg_num_workloads.csv')
    
    violinplot_overall_avg(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_overall_worst(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_time_slices_avg(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_time_slices_worst(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_time_slices_avg_only_demand(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_time_slices_worst_only_demand(df, output_dir=f'{fair_co2_path}/figures')
    scatter_num_workloads_avg(df, p95_num_workloads_df, avg_num_workloads_df, output_dir=f'{fair_co2_path}/figures')
    scatter_num_workloads_worst(df, p95_num_workloads_df, avg_num_workloads_df, output_dir=f'{fair_co2_path}/figures')


if __name__ == '__main__':
    main()