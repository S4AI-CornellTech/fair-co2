import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
import os

# Set global style
figsize = (6, 4)
xlabel_fontsize = 22
ylabel_fontsize = 22
tick_fontsize = 20
title_fontsize = 20
legend_fontsize = 18.5
dpi = 200
palette = "Dark2"
scatter_marker_size = 20
scatter_alpha = 0.3
legend_markersize = 20
legend_alpha = 1
edgewidth = 0
linewidth = 1
violin_width = 0.8
baseline_name = 'RUP-Baseline'
adjusted_name = 'FAIR-CO2'

fair_co2_path = os.environ.get('FAIR_CO2')

# Fig box and whisker comparison avg and max deviation from shapley attributions for different methods
def violinplot_all_values(df_val, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = df_val[df_val['Attribution Method'] == adjusted_name]
    vp = sns.violinplot(
        data=df_filtered, x="Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=.6, palette=palette, ax=ax, linewidth=linewidth
    )
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/all_values_violin_absolute_samples.pdf'
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

def violinplot_all_values_absolute_samples(df_val, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df = df_val.copy()
    df['Deviation (%)'] = np.abs(df['Deviation (%)'])
    df_filtered = df[df['Attribution Method'] == adjusted_name]
    vp = sns.violinplot(
        data=df_filtered, x="Num Samples", y="Deviation (%)", hue="Attribution Method",
        width=.6, palette=palette, ax=ax, linewidth=linewidth
    )
    vp.set(ylabel=None)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/all_values_violin_absolute_samples.pdf'
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
    # Plot one box for each attribution method on the same plot
    vp = sns.violinplot(
        data=df, x="Average Deviation (%)", y="Attribution Method", hue="Attribution Method",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    vp.set(ylabel=None)
    # Remove y-axis ticks
    vp.set(yticklabels=[])
    vp.tick_params(left=False)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # Add legend with color and labels for each attribution method
    plt.legend(bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes, \
               fontsize=legend_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8a_overall_deviation_violin_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_workload(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == baseline_name]
    vp = sns.violinplot(
        data=df_filtered, x="Workload", y="Deviation (%)", hue="Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order
    )
    plt.xlabel('Workload', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    ticks = np.arange(0, 15, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels, ticks=ticks)
    vp.set(xlabel=None)
    plt.xlabel('Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(baseline_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/per_workload_deviation_violin.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_workload_adjusted(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == adjusted_name]
    vp = sns.violinplot(
        data=df_filtered, x="Workload", y="Deviation (%)", hue="Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order
    )
    plt.xlabel('Workload', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.ylim(-15, 15)
    # plt.yticks(fontsize=tick_fontsize)
    ticks = np.arange(0, 15, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels, ticks=ticks)
    vp.set(xlabel=None)
    plt.xlabel('Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(adjusted_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/per_workload_deviation_violin_adjusted.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_workload_absolute(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == baseline_name]
    df_filtered['Deviation (%)'] = np.abs(df_filtered['Deviation (%)'])
    vp = sns.violinplot(
        data=df_filtered, x="Workload", y="Deviation (%)", hue="Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order
    )
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(-0.5, 47)
    ticks = np.arange(0, 15, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels, ticks=ticks)
    vp.set(xlabel=None)
    plt.xlabel('Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(baseline_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/9a_per_workload_deviation_violin_absolute.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_workload_absolute_adjusted(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == adjusted_name]
    df_filtered['Deviation (%)'] = np.abs(df_filtered['Deviation (%)'])
    vp = sns.violinplot(
        data=df_filtered, x="Workload", y="Deviation (%)", hue="Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order
    )
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.ylim(-0.5, 47)
    plt.yticks(fontsize=tick_fontsize)
    ticks = np.arange(0, 15, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels, ticks=ticks)
    vp.set(xlabel=None)
    plt.xlabel('Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(adjusted_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/9b_per_workload_deviation_violin_absolute_adjusted.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_partner_workload(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == baseline_name]
    vp = sns.violinplot(
        data=df_filtered, x="Partner Workload", y="Deviation (%)", hue="Partner Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order + ['nothing']
    )
    plt.xlabel('Workload', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize-4)
    plt.xlabel('Partner Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(baseline_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/per_partner_workload_deviation_violin.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_partner_workload_absolute(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == baseline_name]
    df_filtered['Deviation (%)'] = np.abs(df_filtered['Deviation (%)'])
    vp = sns.violinplot(
        data=df_filtered, x="Partner Workload", y="Deviation (%)", hue="Partner Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order + ['nothing']
    )
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.ylim(-0.5, 47)
    plt.yticks(fontsize=tick_fontsize)
    ticks = np.arange(0, 16, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels + ['NONE'], ticks=ticks)
    vp.set(xlabel=None)
    # Shift x label up
    plt.xlabel('Partner Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(baseline_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/9c_per_partner_workload_deviation_violin_absolute.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_per_partner_workload_absolute_adjusted(df, output_dir, order, labels):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot one box for each attribution method on the same plot
    df_filtered = df[df['Attribution Method'] == adjusted_name]
    df_filtered['Deviation (%)'] = np.abs(df_filtered['Deviation (%)'])
    vp = sns.violinplot(
        data=df_filtered, x="Partner Workload", y="Deviation (%)", hue="Partner Workload",
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend=False, order=order + ['nothing']
    )
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(-0.5, 47)
    ticks = np.arange(0, 16, 1)
    plt.xticks(fontsize=tick_fontsize-4, rotation=90, ha='center', labels=labels + ['NONE'], ticks=ticks)
    vp.set(xlabel=None)
    plt.xlabel('Partner Workload', fontsize=xlabel_fontsize, labelpad=0)
    plt.title(adjusted_name, fontsize=title_fontsize, pad=5)
    plt.tight_layout()
    output_file = f'{output_dir}/9d_per_partner_workload_deviation_violin_absolute_adjusted.pdf'
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
        width=0.9, palette=palette, ax=ax, linewidth=linewidth, legend='full'
    )
    vp.set(ylabel=None)
    vp.set(yticklabels=[])
    vp.tick_params(left=False)
    plt.xlabel('Deviation (%)', fontsize=xlabel_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.ylabel('Attribution Method', fontsize=14)
    # plt.title(f'Worst-Case Deviation', fontsize=title_fontsize)
    plt.legend(bbox_to_anchor=(1, 0.56), bbox_transform=ax.transAxes, \
            fontsize=legend_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8e_overall_deviation_violin_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

# fig box and whisker for different samples avg and max deviation from shapley attributions
def box_and_whisker_samples_avg(interference_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = interference_data_df[interference_data_df['Attribution Method'] == adjusted_name]
    bp = sns.boxplot(
        data=df_filtered, x="Num Samples", y="Average Deviation (%)", hue="Attribution Method",
        whis=[0, 100], width=.6, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Samples', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/n_samples_deviation_box_and_whisker_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_samples_avg(interference_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = interference_data_df[interference_data_df['Attribution Method'] == adjusted_name]
    df_filtered_baseline = interference_data_df[interference_data_df['Attribution Method'] == baseline_name]
    df_filtered_baseline['Num Samples'] = 0
    df_filtered = pd.concat([df_filtered, df_filtered_baseline])
    vp = sns.violinplot(
        data=df_filtered, x="Num Samples", y="Average Deviation (%)", hue="Attribution Method",
        width=.9, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Samples', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize-6)
    # Remove the tick and label for x = 0
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # Shift y label to right
    ax.xaxis.set_label_coords(0.55, -0.13)
    # Set y lim
    ax.set_ylim(-0.5, 12.5)
    plt.legend(fontsize=legend_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8b_n_samples_deviation_violin_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def box_and_whisker_samples_worst(interference_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = interference_data_df[interference_data_df['Attribution Method'] == adjusted_name]
    sns.boxplot(
        data=df_filtered, x="Num Samples", y="Worst-Case Deviation (%)",
        whis=[0, 100], width=violin_width, ax=ax, legend=False, linewidth=linewidth
    )
    plt.xlabel('Number of Samples', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/n_samples_deviation_box_and_whisker_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def violinplot_samples_worst(interference_data_df, output_dir):
    fig, ax = plt.subplots(figsize=figsize)
    # Ignore num_workloads and num_time_slices
    # Plot one box for each attribution method on the same plot
    df_filtered = interference_data_df[interference_data_df['Attribution Method'] == adjusted_name]
    df_filtered_baseline = interference_data_df[interference_data_df['Attribution Method'] == baseline_name]
    df_filtered_baseline['Num Samples'] = 0
    df_filtered = pd.concat([df_filtered, df_filtered_baseline])
    vp = sns.violinplot(
        data=df_filtered, x="Num Samples", y="Worst-Case Deviation (%)", hue='Attribution Method',
        width=.9, palette=palette, ax=ax, linewidth=linewidth,
    )
    plt.xlabel('Number of Samples', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize-6)
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # Shift y label to right
    ax.xaxis.set_label_coords(0.55, -0.13)
    ax.set_ylim(-0.5, 37)
    plt.legend(fontsize=legend_fontsize)
    #plt.title(f' Average Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8f_n_samples_deviation_violin_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

# Avg and max dev scatter over grid ci
def scatter_grid_ci_avg(df, p95_df, avg_df, output_dir):
    plt.figure(figsize=figsize)
    # Plot one scatter for each attribution method on the same plot
    sp = sns.scatterplot(
        data=df, x="Grid CI", y="Average Deviation (%)", hue="Attribution Method",
        palette=palette, s=scatter_marker_size, alpha=scatter_alpha, linewidth=edgewidth
    )
    # The 95th percentile of deviation for each method as a line
    
    handles, labels = sp.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(legend_markersize)
        h.set_alpha(legend_alpha)

    # Plot the 95th percentile of deviation for each method
    p95_baseline_df = p95_df[p95_df['Attribution Method'] == baseline_name]
    p95_adjusted_df = p95_df[p95_df['Attribution Method'] == adjusted_name]
    sp.plot(p95_baseline_df['Grid CI'], p95_baseline_df['95th Percentile Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_adjusted_df['Grid CI'], p95_adjusted_df['95th Percentile Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_adjusted_df = avg_df[avg_df['Attribution Method'] == adjusted_name]

    sp.plot(avg_baseline_df['Grid CI'], avg_baseline_df['Average Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_adjusted_df['Grid CI'], avg_adjusted_df['Average Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    # legend as 2 columns
    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)
    plt.xlabel('Grid Carbon Intensity ($gCO_{2}e/kWh$)', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.title(f'Grid CI vs Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8d_grid_ci_deviation_scatter_avg.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def scatter_grid_ci_worst(df, p95_df, avg_df, output_dir):
    plt.figure(figsize=figsize)
    # Plot one scatter for each attribution method on the same plot
    sp = sns.scatterplot(
        data=df, x="Grid CI", y="Worst-Case Deviation (%)", hue="Attribution Method",
        palette=palette, s=scatter_marker_size, alpha=scatter_alpha, linewidth=edgewidth
    )
    handles, labels = sp.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(legend_markersize)
        h.set_alpha(legend_alpha)

        # Plot the 95th percentile of deviation for each method
    p95_baseline_df = p95_df[p95_df['Attribution Method'] == baseline_name]
    p95_adjusted_df = p95_df[p95_df['Attribution Method'] == adjusted_name]
    sp.plot(p95_baseline_df['Grid CI'], p95_baseline_df['95th Percentile Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_adjusted_df['Grid CI'], p95_adjusted_df['95th Percentile Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_adjusted_df = avg_df[avg_df['Attribution Method'] == adjusted_name]

    sp.plot(avg_baseline_df['Grid CI'], avg_baseline_df['Average Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_adjusted_df['Grid CI'], avg_adjusted_df['Average Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)
    plt.ylim(0, 81)
    plt.xlabel('Grid Carbon Intensity ($gCO_{2}e/kWh$)', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.title(f'Grid CI vs Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8h_grid_ci_deviation_scatter_worst.pdf'
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
    p95_adjusted_df = p95_df[p95_df['Attribution Method'] == adjusted_name]
    sp.plot(p95_baseline_df['Num Workloads'], p95_baseline_df['95th Percentile Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_adjusted_df['Num Workloads'], p95_adjusted_df['95th Percentile Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_adjusted_df = avg_df[avg_df['Attribution Method'] == adjusted_name]

    sp.plot(avg_baseline_df['Num Workloads'], avg_baseline_df['Average Average Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_adjusted_df['Num Workloads'], avg_adjusted_df['Average Average Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)

    plt.ylim(0, 25)

    plt.xlabel('Number of Workloads', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)

    plt.tight_layout()
    output_file = f'{output_dir}/8c_num_workloads_deviation_scatter_avg.pdf'
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
    p95_adjusted_df = p95_df[p95_df['Attribution Method'] == adjusted_name]
    sp.plot(p95_baseline_df['Num Workloads'], p95_baseline_df['95th Percentile Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5, linestyle='--')
    sp.plot(p95_adjusted_df['Num Workloads'], p95_adjusted_df['95th Percentile Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5, linestyle='--')
    
    # Plot the average deviation for each method
    avg_baseline_df = avg_df[avg_df['Attribution Method'] == baseline_name]
    avg_adjusted_df = avg_df[avg_df['Attribution Method'] == adjusted_name]

    sp.plot(avg_baseline_df['Num Workloads'], avg_baseline_df['Average Worst-Case Deviation (%)'], 
            color='darkslategray', linewidth=linewidth*1.5)
    sp.plot(avg_adjusted_df['Num Workloads'], avg_adjusted_df['Average Worst-Case Deviation (%)'],
            color='darkred', linewidth=linewidth*1.5)
    
    plt.plot([], [], color='black', linewidth=linewidth*1.5, linestyle='--', label='P95')
    plt.plot([], [], color='black', linewidth=linewidth*1.5, label='Mean')

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=2)
    plt.ylim(0, 71)
    plt.xlabel('Number of Workloads', fontsize=xlabel_fontsize)
    plt.ylabel('Deviation (%)', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.title(f'Grid CI vs Deviation', fontsize=title_fontsize)
    plt.tight_layout()
    output_file = f'{output_dir}/8g_num_workloads_deviation_scatter_worst.pdf'
    print(f'Saving figure to {output_file}')
    plt.savefig(output_file, dpi=dpi)
    return

def process_interference_data_chunk(interference_data_df_chunk):
    df_chunk = pd.DataFrame(columns=['Attribution Method', 'Average Deviation (%)', 'Worst-Case Deviation (%)', 'Num Workloads', 'Num Samples', 'Grid CI'])
    num_simulations = len(interference_data_df_chunk)
    # Process
    for i, row in interference_data_df_chunk.iterrows():
        df_chunk = df_chunk._append({'Attribution Method': baseline_name, 
                        'Average Deviation (%)': row['baseline_deviation_from_shapley (%)'], 
                        'Worst-Case Deviation (%)': row['worst_case_baseline_deviation (%)'],
                        'Num Workloads': row['n_workloads'],
                        'Num Samples': row['num_samples'],
                        'Grid CI': row['grid_ci']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': adjusted_name, 
                        'Average Deviation (%)': row['adjusted_deviation_from_shapley (%)'], 
                        'Worst-Case Deviation (%)': row['worst_case_adjusted_deviation (%)'],
                        'Num Workloads': row['n_workloads'],
                        'Num Samples': row['num_samples'],
                        'Grid CI': row['grid_ci']}, ignore_index=True)
    return df_chunk

def process_interference_data(interference_data_df_file, num_workers):
    interference_data_df = pd.read_csv(interference_data_df_file)
    interference_data_df = interference_data_df.astype({'n_workloads': 'int32', 
                                                        'num_samples': 'int32', 
                                                        'grid_ci': 'float64',
                                                        'baseline_deviation_from_shapley (%)': 'float64', 
                                                        'adjusted_deviation_from_shapley (%)': 'float64', 
                                                        'worst_case_baseline_deviation (%)': 'float64', 
                                                        'worst_case_adjusted_deviation (%)': 'float64'})
    df = pd.DataFrame(columns=['Attribution Method', 'Average Deviation (%)', 'Worst-Case Deviation (%)', 'Num Workloads', 'Num Samples', 'Grid CI'])
    num_simulations = len(interference_data_df)
    # Process in chunks in parallel
    num_chunks = num_workers
    chunk_size = num_simulations // num_chunks
    interference_data_df_chunks = [interference_data_df[i*chunk_size:(i+1)*chunk_size] for i in range(0, num_chunks)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        df_chunks = executor.map(process_interference_data_chunk, interference_data_df_chunks)
    df = pd.concat(df_chunks)
    df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_processed.csv')

def process_interference_val_data_chunk(interference_val_data_df_chunk):
    df_chunk = pd.DataFrame(columns=['Attribution Method', 'Deviation (%)', 'Workload', 'Partner Workload', 'Num Workloads', 'Num Samples', 'Grid CI'])
    num_simulations = len(interference_val_data_df_chunk)
    for i, row in interference_val_data_df_chunk.iterrows():
        df_chunk = df_chunk._append({'Attribution Method': baseline_name, 
                        'Deviation (%)': row['baseline_deviation (%)'], 
                        'Workload': row['workload'],
                        'Partner Workload': row['partner_workload'],
                        'Num Workloads': row['n_workloads'],
                        'Num Samples': row['num_samples'],
                        'Grid CI': row['grid_ci']}, ignore_index=True)
        df_chunk = df_chunk._append({'Attribution Method': adjusted_name, 
                        'Deviation (%)': row['adjusted_deviation (%)'], 
                        'Workload': row['workload'],
                        'Partner Workload': row['partner_workload'],
                        'Num Workloads': row['n_workloads'],
                        'Num Samples': row['num_samples'],
                        'Grid CI': row['grid_ci']}, ignore_index=True)
    return df_chunk

def process_interference_val_data(interference_val_data_df_file, num_workers=10):
    interference_data_df = pd.read_csv(interference_val_data_df_file)
    interference_data_df = interference_data_df.astype({'n_workloads': 'int32', 
                                                        'num_samples': 'int32', 
                                                        'grid_ci': 'float64',
                                                        'workload': 'str',
                                                        'partner_workload': 'str',
                                                        'baseline_deviation (%)': 'float64', 
                                                        'adjusted_deviation (%)': 'float64'})
    df = pd.DataFrame(columns=['Attribution Method', 'Deviation (%)', 'Workload', 'Partner Workload', 'Num Workloads', 'Num Samples', 'Grid CI'])
    num_chunks = num_workers
    chunk_size = len(interference_data_df) // num_chunks
    interference_data_df_chunks = [interference_data_df[i*chunk_size:(i+1)*chunk_size] for i in range(0, num_chunks)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        df_chunks = executor.map(process_interference_val_data_chunk, interference_data_df_chunks)
    df = pd.concat(df_chunks)
    df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_val_processed.csv')

def process_p95_grid_ci_data(df):
    # For each grid ci range of 10, find the 95th percentile of deviation for each attribution method
    grid_ci_min = df['Grid CI'].min()
    grid_ci_max = df['Grid CI'].max()
    grid_ci_ranges = np.arange(grid_ci_min, grid_ci_max + 1, 10)
    p95_df = pd.DataFrame(columns=['Attribution Method', 'Grid CI', '95th Percentile Average Deviation (%)', '95th Percentile Worst-Case Deviation (%)'])
    p95_avg_baseline_dev_list = []
    p95_avg_adjusted_dev_list = []
    p95_worst_baseline_dev_list = []
    p95_worst_adjusted_dev_list = []
    for i in range(len(grid_ci_ranges) - 1):
        lower_bound = grid_ci_ranges[i]
        upper_bound = grid_ci_ranges[i+1]
        df_filtered = df[(df['Grid CI'] >= lower_bound) & (df['Grid CI'] < upper_bound)]
        p95_avg_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'], 95)
        p95_avg_adjusted_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Average Deviation (%)'], 95)
        p95_worst_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'], 95)
        p95_worst_adjusted_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Worst-Case Deviation (%)'], 95)
        p95_avg_baseline_dev_list.append(p95_avg_baseline_dev)
        p95_avg_adjusted_dev_list.append(p95_avg_adjusted_dev)
        p95_worst_baseline_dev_list.append(p95_worst_baseline_dev)
        p95_worst_adjusted_dev_list.append(p95_worst_adjusted_dev)
    
    # Apply moving average to smooth the curve
    sigma = 10
    p95_avg_baseline_dev_list = gaussian_filter1d(p95_avg_baseline_dev_list, sigma=sigma)
    p95_avg_adjusted_dev_list = gaussian_filter1d(p95_avg_adjusted_dev_list, sigma=sigma)
    p95_worst_baseline_dev_list = gaussian_filter1d(p95_worst_baseline_dev_list, sigma=sigma)
    p95_worst_adjusted_dev_list = gaussian_filter1d(p95_worst_adjusted_dev_list, sigma=sigma)

    # Add to p95_df
    for i in range(len(p95_avg_baseline_dev_list)):
        p95_df = p95_df._append({'Attribution Method': baseline_name, 
                                'Grid CI': grid_ci_ranges[i], 
                                '95th Percentile Average Deviation (%)': p95_avg_baseline_dev_list[i],
                                '95th Percentile Worst-Case Deviation (%)': p95_worst_baseline_dev_list[i]}, ignore_index=True)
        p95_df = p95_df._append({'Attribution Method': adjusted_name, 
                                'Grid CI': grid_ci_ranges[i], 
                                '95th Percentile Average Deviation (%)': p95_avg_adjusted_dev_list[i],
                                '95th Percentile Worst-Case Deviation (%)': p95_worst_adjusted_dev_list[i]}, ignore_index=True)
    p95_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_grid_ci.csv')
    return p95_df

def process_avg_grid_ci_data(df):
    # For each grid ci range of 10, find the average deviation for each attribution method
    grid_ci_min = df['Grid CI'].min()
    grid_ci_max = df['Grid CI'].max()
    grid_ci_ranges = np.arange(grid_ci_min, grid_ci_max + 1, 10)
    avg_df = pd.DataFrame(columns=['Attribution Method', 'Grid CI', 'Average Average Deviation (%)', 'Average Worst-Case Deviation (%)'])
    avg_avg_baseline_dev_list = []
    avg_avg_adjusted_dev_list = []
    avg_worst_baseline_dev_list = []
    avg_worst_adjusted_dev_list = []
    for i in range(len(grid_ci_ranges) - 1):
        lower_bound = grid_ci_ranges[i]
        upper_bound = grid_ci_ranges[i+1]
        df_filtered = df[(df['Grid CI'] >= lower_bound) & (df['Grid CI'] < upper_bound)]
        avg_avg_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'])
        avg_avg_adjusted_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Average Deviation (%)'])
        avg_worst_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'])
        avg_worst_adjusted_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Worst-Case Deviation (%)'])
        avg_avg_baseline_dev_list.append(avg_avg_baseline_dev)
        avg_avg_adjusted_dev_list.append(avg_avg_adjusted_dev)
        avg_worst_baseline_dev_list.append(avg_worst_baseline_dev)
        avg_worst_adjusted_dev_list.append(avg_worst_adjusted_dev)
    
    # Apply moving average to smooth the curve
    sigma = 10
    avg_avg_baseline_dev_list = gaussian_filter1d(avg_avg_baseline_dev_list, sigma=sigma)
    avg_avg_adjusted_dev_list = gaussian_filter1d(avg_avg_adjusted_dev_list, sigma=sigma)
    avg_worst_baseline_dev_list = gaussian_filter1d(avg_worst_baseline_dev_list, sigma=sigma)
    avg_worst_adjusted_dev_list = gaussian_filter1d(avg_worst_adjusted_dev_list, sigma=sigma)

    # Add to avg_df
    for i in range(len(avg_avg_baseline_dev_list)):
        avg_df = avg_df._append({'Attribution Method': baseline_name, 
                                'Grid CI': grid_ci_ranges[i], 
                                'Average Average Deviation (%)':avg_avg_baseline_dev_list[i],
                                'Average Worst-Case Deviation (%)': avg_worst_baseline_dev_list[i]}, ignore_index=True)
        avg_df = avg_df._append({'Attribution Method': adjusted_name,
                                'Grid CI': grid_ci_ranges[i], 
                                'Average Average Deviation (%)': avg_avg_adjusted_dev_list[i],
                                'Average Worst-Case Deviation (%)': avg_worst_adjusted_dev_list[i]}, ignore_index=True)
    avg_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_grid_ci.csv')
    return avg_df

def process_p95_num_workloads_data(df):
    # For each num workloads, find the 95th percentile of deviation for each attribution method
    min_num_workloads = df['Num Workloads'].min()
    max_num_workloads = df['Num Workloads'].max()
    num_workloads = np.arange(min_num_workloads, max_num_workloads + 1, 1)
    p95_df = pd.DataFrame(columns=['Attribution Method', 'Num Workloads', '95th Percentile Average Deviation (%)', '95th Percentile Worst-Case Deviation (%)'])
    p95_avg_baseline_dev_list = []
    p95_avg_adjusted_dev_list = []
    p95_worst_baseline_dev_list = []
    p95_worst_adjusted_dev_list = []
    for i in range(len(num_workloads)):
        num_workload = num_workloads[i]
        df_filtered = df[df['Num Workloads'] == num_workload]
        print(f'Num workloads: {num_workload}, num samples: {len(df_filtered)}')
        p95_avg_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'], 95)
        p95_avg_adjusted_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Average Deviation (%)'], 95)
        p95_worst_baseline_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'], 95)
        p95_worst_adjusted_dev = np.percentile(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Worst-Case Deviation (%)'], 95)
        p95_avg_baseline_dev_list.append(p95_avg_baseline_dev)
        p95_avg_adjusted_dev_list.append(p95_avg_adjusted_dev)
        p95_worst_baseline_dev_list.append(p95_worst_baseline_dev)
        p95_worst_adjusted_dev_list.append(p95_worst_adjusted_dev)
    
    # Apply moving average to smooth the curve
    sigma = 5
    p95_avg_baseline_dev_list = gaussian_filter1d(p95_avg_baseline_dev_list, sigma=sigma)
    p95_avg_adjusted_dev_list = gaussian_filter1d(p95_avg_adjusted_dev_list, sigma=sigma)
    p95_worst_baseline_dev_list = gaussian_filter1d(p95_worst_baseline_dev_list, sigma=sigma)
    p95_worst_adjusted_dev_list = gaussian_filter1d(p95_worst_adjusted_dev_list, sigma=sigma)

    # Add to p95_df
    for i in range(len(p95_avg_baseline_dev_list)):
        p95_df = p95_df._append({'Attribution Method': baseline_name,
                                 'Num Workloads': num_workloads[i], 
                                '95th Percentile Average Deviation (%)': p95_avg_baseline_dev_list[i],
                                '95th Percentile Worst-Case Deviation (%)': p95_worst_baseline_dev_list[i]}, ignore_index=True)
        p95_df = p95_df._append({'Attribution Method': adjusted_name,
                                    'Num Workloads': num_workloads[i], 
                                    '95th Percentile Average Deviation (%)': p95_avg_adjusted_dev_list[i],
                                    '95th Percentile Worst-Case Deviation (%)': p95_worst_adjusted_dev_list[i]}, ignore_index=True)
    p95_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_num_workloads.csv')
    return p95_df

def process_avg_num_workloads_data(df):
    # For each num workloads, find the average deviation for each attribution method
    min_num_workloads = df['Num Workloads'].min()
    max_num_workloads = df['Num Workloads'].max()
    num_workloads = np.arange(min_num_workloads, max_num_workloads + 1, 1)
    avg_df = pd.DataFrame(columns=['Attribution Method', 'Num Workloads', 'Average Average Deviation (%)', 'Average Worst-Case Deviation (%)'])
    avg_avg_baseline_dev_list = []
    avg_avg_adjusted_dev_list = []
    avg_worst_baseline_dev_list = []
    avg_worst_adjusted_dev_list = []
    for i in range(len(num_workloads)):
        num_workload = num_workloads[i]
        df_filtered = df[df['Num Workloads'] == num_workload]
        avg_avg_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Average Deviation (%)'])
        avg_avg_adjusted_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Average Deviation (%)'])
        avg_worst_baseline_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'])
        avg_worst_adjusted_dev = np.mean(df_filtered[df_filtered['Attribution Method'] == adjusted_name]['Worst-Case Deviation (%)'])
        avg_avg_baseline_dev_list.append(avg_avg_baseline_dev)
        avg_avg_adjusted_dev_list.append(avg_avg_adjusted_dev)
        avg_worst_baseline_dev_list.append(avg_worst_baseline_dev)
        avg_worst_adjusted_dev_list.append(avg_worst_adjusted_dev)

    # Apply moving average to smooth the curve
    sigma = 5
    avg_avg_baseline_dev_list = gaussian_filter1d(avg_avg_baseline_dev_list, sigma=sigma)
    avg_avg_adjusted_dev_list = gaussian_filter1d(avg_avg_adjusted_dev_list, sigma=sigma)
    avg_worst_baseline_dev_list = gaussian_filter1d(avg_worst_baseline_dev_list, sigma=sigma)
    avg_worst_adjusted_dev_list = gaussian_filter1d(avg_worst_adjusted_dev_list, sigma=sigma)

    # Add to avg_df
    for i in range(len(avg_avg_baseline_dev_list)):
        avg_df = avg_df._append({'Attribution Method': baseline_name,
                                 'Num Workloads': num_workloads[i], 
                                'Average Average Deviation (%)': avg_avg_baseline_dev_list[i],
                                'Average Worst-Case Deviation (%)': avg_worst_baseline_dev_list[i]}, ignore_index=True)
        avg_df = avg_df._append({'Attribution Method': adjusted_name,
                                    'Num Workloads': num_workloads[i], 
                                    'Average Average Deviation (%)': avg_avg_adjusted_dev_list[i],
                                    'Average Worst-Case Deviation (%)': avg_worst_adjusted_dev_list[i]}, ignore_index=True)
    avg_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_num_workloads.csv')
    return avg_df


def main():
    num_workers = 20
    process_interference_data(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results.csv', num_workers=num_workers)
    process_interference_val_data(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_vals.csv', num_workers=num_workers)
    workload_order = [
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
    labels = [
        'DDUP',
        'BFS',
        'MSF',
        'WC',
        'SA',
        'CH',
        'NN',
        'NBODY',
        'PG-100',
        'PG-50',
        'PG-10',
        'H.265',
        'LLAMA',
        'FAISS',
        'SPARK'
    ]
    df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_processed.csv')
    df = df.astype({'Num Workloads': 'int32', 
                    'Num Samples': 'int32', 
                    'Grid CI': 'float64',
                    'Average Deviation (%)': 'float64', 
                    'Worst-Case Deviation (%)': 'float64'})
    df_val = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_val_processed.csv')
    df_val = df_val.astype({'Num Workloads': 'int32', 
                            'Num Samples': 'int32', 
                            'Grid CI': 'float64',
                            'Deviation (%)': 'float64'})
    p95_grid_ci_df = process_p95_grid_ci_data(df)
    avg_grid_ci_df = process_avg_grid_ci_data(df)
    p95_num_workloads_df = process_p95_num_workloads_data(df)
    avg_num_workloads_df = process_avg_num_workloads_data(df)

    # Save p95 and avg dataframes
    p95_grid_ci_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_grid_ci.csv')
    avg_grid_ci_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_grid_ci.csv')
    p95_num_workloads_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_num_workloads.csv')
    avg_num_workloads_df.to_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_num_workloads.csv')

    # Read p95 and avg dataframes
    # p95_grid_ci_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_grid_ci.csv')
    # avg_grid_ci_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_grid_ci.csv')
    # p95_num_workloads_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_p95_num_workloads.csv')
    # avg_num_workloads_df = pd.read_csv(f'{fair_co2_path}/monte-carlo-simulations/colocation/sim-results/interference_adjustment_results_avg_num_workloads.csv')
    
    avg_avg_deviation = np.mean(df[df['Attribution Method'] == adjusted_name]['Average Deviation (%)'])
    avg_worst_deviation = np.mean(df[df['Attribution Method'] == adjusted_name]['Worst-Case Deviation (%)'])
    print('Adjusted average deviation for average case:', avg_avg_deviation)
    print('Adjusted average deviation for worst case:', avg_worst_deviation)

    avg_avg_deviation = np.mean(df[df['Attribution Method'] == baseline_name]['Average Deviation (%)'])
    avg_worst_deviation = np.mean(df[df['Attribution Method'] == baseline_name]['Worst-Case Deviation (%)'])
    print('Baseline average deviation for average case:', avg_avg_deviation)
    print('Baseline average deviation for worst case:', avg_worst_deviation)

    violinplot_overall_avg(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_overall_worst(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_samples_avg(df, output_dir=f'{fair_co2_path}/figures')
    violinplot_samples_worst(df, output_dir=f'{fair_co2_path}/figures')
    scatter_grid_ci_avg(df, p95_grid_ci_df, avg_grid_ci_df, output_dir=f'{fair_co2_path}/figures')
    scatter_grid_ci_worst(df, p95_grid_ci_df, avg_grid_ci_df, output_dir=f'{fair_co2_path}/figures')
    scatter_num_workloads_avg(df, p95_num_workloads_df, avg_num_workloads_df, output_dir=f'{fair_co2_path}/figures')
    scatter_num_workloads_worst(df, p95_num_workloads_df, avg_num_workloads_df, output_dir=f'{fair_co2_path}/figures')
    violinplot_per_workload_absolute(df_val, output_dir=f'{fair_co2_path}/figures', order=workload_order, labels=labels)
    violinplot_per_partner_workload_absolute(df_val, output_dir=f'{fair_co2_path}/figures', order=workload_order, labels=labels)
    violinplot_per_partner_workload_absolute_adjusted(df_val, output_dir=f'{fair_co2_path}/figures', order=workload_order, labels=labels)
    violinplot_per_workload_absolute_adjusted(df_val, output_dir=f'{fair_co2_path}/figures', order=workload_order, labels=labels)


if __name__ == '__main__':
    main()