import itertools
import math
import numpy as np
import pandas as pd

def peak_finder(players_list, player_peaks):
    peak = 0
    for player in players_list:
        player_peak = player_peaks[player]
        if player_peak > peak:
            peak = player_peak
    return peak

def peak_shapley(players_list, player_peaks, player_resource_times, attribution_total):
    num_players = len(players_list)
    player_combinations = []
    for k in range(0, num_players + 1):
        player_combinations += list(itertools.combinations(players_list, k))
    for i in range(0, len(player_combinations)):
        player_combinations[i] = set(player_combinations[i])
    shapley_attributions = []
    for player in players_list:
        shapley_value = 0
        for combination in player_combinations:
            if player in combination:
                combination_without_player = combination - {player}
                peak_without_player = peak_finder(combination_without_player, player_peaks)
                peak_with_player = peak_finder(combination, player_peaks)
                scaling_factor = 1 / math.comb(num_players-1, len(combination_without_player))
                shapley_value += (peak_with_player - peak_without_player) * scaling_factor 

        shapley_attributions.append(shapley_value * player_resource_times[player])
    shapley_attribution_sum = sum(shapley_attributions)
    if shapley_attribution_sum == 0:
        shapley_attribution_totals = [0 for player in players_list]
        # print(len(shapley_attribution_totals))
        return shapley_attribution_totals
    else:
        shapley_ratios = [attribution / shapley_attribution_sum for attribution in shapley_attributions]
        shapley_attribution_totals = [shapley_ratio * attribution_total for shapley_ratio in shapley_ratios]
        # print(len(shapley_attribution_totals))
        return shapley_attribution_totals

def peaks(timeseries_data, time_column_name, value_column_name, time_granularities):
    if (len(time_granularities) == 0): # base case, time_granularities is empty 
        return [timeseries_data[value_column_name].max()]
    else:
        results = []
        peaks_list = []
        time_granularity = time_granularities[0]
        min_time = timeseries_data[time_column_name].min()
        n_time_blocks = math.ceil((timeseries_data[time_column_name].max() - min_time + 1) / time_granularity)
        for block in range(0, n_time_blocks - 1):
            time_block = timeseries_data[(timeseries_data[time_column_name] >= (block * time_granularity + min_time)) & 
                                    (timeseries_data[time_column_name] < ((block + 1) * time_granularity) + min_time)]
            block_peaks = peaks(time_block, time_column_name, value_column_name, time_granularities[1:])
            results.append(block_peaks)
            peaks_list.append(block_peaks[0])
        
        # last time block
        time_block = timeseries_data[timeseries_data[time_column_name] >= ((n_time_blocks - 1) * time_granularity + min_time)]
        block_peaks = peaks(time_block, time_column_name, value_column_name, time_granularities[1:])
        results.append(block_peaks)
        peaks_list.append(block_peaks[0])

        # find peak amongst all blocks
        global_peak = max(peaks_list)

        return [global_peak] + results

def resource_time(timeseries_data, time_column_name, value_column_name, time_granularities):
    if (len(time_granularities) == 0):
        return [timeseries_data[value_column_name].sum()]
    else:
        results = []
        resource_time_list = []
        time_granularity = time_granularities[0]
        min_time = timeseries_data[time_column_name].min()
        n_time_blocks = math.ceil((timeseries_data[time_column_name].max() - min_time + 1) / time_granularity)
        for block in range(0, n_time_blocks - 1):
            time_block = timeseries_data[(timeseries_data[time_column_name] >= (block * time_granularity + min_time)) & 
                                    (timeseries_data[time_column_name] < ((block + 1) * time_granularity) + min_time)]
            block_resource_time = resource_time(time_block, time_column_name, value_column_name, time_granularities[1:])
            results.append(block_resource_time)
            resource_time_list.append(block_resource_time[0])
        
        # last time block
        time_block = timeseries_data[timeseries_data[time_column_name] >= ((n_time_blocks - 1) * time_granularity + min_time)]
        block_resource_time = resource_time(time_block, time_column_name, value_column_name, time_granularities[1:])
        results.append(block_resource_time)
        resource_time_list.append(block_resource_time[0])

        # find peak amongst all blocks
        global_resource_time = sum(resource_time_list)

        return [global_resource_time] + results

def hierarchical_shapley(hierarchical_peaks, hierarchical_resource_times, attribution_total):
    players_list = range(0, len(hierarchical_peaks) - 1)
    # print(hierarchical_peaks)
    player_peaks = []
    player_resource_times = []
    for player in players_list:
        player_peaks.append(hierarchical_peaks[player + 1][0])
        player_resource_times.append(hierarchical_resource_times[player + 1][0])
    shapley_values = peak_shapley(players_list=players_list, player_peaks=player_peaks, player_resource_times=player_resource_times, attribution_total=attribution_total)

    if len(hierarchical_peaks[1]) <= 1:
        return [shapley_values], [player_peaks], [player_resource_times]

    else:
        lower_level_shapley = []
        lower_level_peaks = []
        for player in players_list:
            player_shapley, player_peak, player_resource_time = hierarchical_shapley(hierarchical_peaks[player + 1], hierarchical_resource_times[player + 1], shapley_values[player])
            if lower_level_shapley == []:
                lower_level_shapley = player_shapley
                lower_level_peaks = player_peak
                lower_level_resource_times = player_resource_time
            else: 
                for i in range(0, len(lower_level_shapley)):
                    lower_level_shapley[i] = np.concatenate((lower_level_shapley[i], player_shapley[i]))
                    lower_level_peaks[i] = np.concatenate((lower_level_peaks[i], player_peak[i]))
                    lower_level_resource_times[i] = np.concatenate((lower_level_resource_times[i], player_resource_time[i]))

        shapley_return = [shapley_values] + lower_level_shapley
        peaks_return = [player_peaks] + lower_level_peaks
        resource_times_return = [player_resource_times] + lower_level_resource_times

        return shapley_return, peaks_return, resource_times_return
    
def carbon_intensity(timeseries_data, time_column_name, value_column_name, time_granularities, shapley_attributions, sampling_interval):
    carbon_intensity = []
    resource_time = []
    for i in range(0, len(time_granularities)):
        time_granularity = time_granularities[i]
        shapley_values = shapley_attributions[i]
        n_time_blocks = len(shapley_values)
        period_carbon_intensity = []
        period_resource_time = []
        for block in range(0, n_time_blocks - 1):
            time_block = timeseries_data[(timeseries_data[time_column_name] >= (block * time_granularity)) & 
                                    (timeseries_data[time_column_name] < ((block + 1) * time_granularity))]
            block_resource_time = time_block[value_column_name].sum() * sampling_interval
            if block_resource_time == 0:
                block_carbon_intensity = 0
            else:
                block_carbon_intensity = shapley_values[block] / block_resource_time
            period_carbon_intensity.append(block_carbon_intensity)
            period_resource_time.append(block_resource_time)
        # last time block
        block = n_time_blocks - 1
        time_block = timeseries_data[timeseries_data[time_column_name] >= (block * time_granularity)]
        block_resource_time = time_block[value_column_name].sum() * sampling_interval
        if block_resource_time == 0:
            block_carbon_intensity = 0
        else:
            block_carbon_intensity = shapley_values[block] / block_resource_time
        period_carbon_intensity.append(block_carbon_intensity)
        period_resource_time.append(block_resource_time)

        carbon_intensity.append(period_carbon_intensity)
        resource_time.append(period_resource_time)

    return carbon_intensity, resource_time
        

# Time has to be in seconds
# Time series data is pandas dataframe
def shapley_attribution(timeseries_data, time_column_name, value_column_name, time_granularities, attribution_total, sampling_interval, offset):

    # Sort time granularities to be largest to smallest
    time_granularities.sort(reverse=True)

    # Offset time_series_data
    timeseries_data = timeseries_data[timeseries_data[time_column_name] >= offset]

    # Find peaks at each time_granularity
    hierarchical_peaks = peaks(timeseries_data=timeseries_data, time_column_name=time_column_name, value_column_name=value_column_name, time_granularities=time_granularities)

    # Find resource time at each time_granularity
    hierarchical_resource_times = resource_time(timeseries_data=timeseries_data, time_column_name=time_column_name, value_column_name=value_column_name, time_granularities=time_granularities)

    # Attribute at each granularity
    shapley, peaks_per_granularity, resource_times_per_granularity = hierarchical_shapley(hierarchical_peaks=hierarchical_peaks, hierarchical_resource_times=hierarchical_resource_times, attribution_total=attribution_total)

    ci, resource_time_val = carbon_intensity(timeseries_data=timeseries_data, time_column_name=time_column_name, value_column_name=value_column_name, time_granularities=time_granularities, shapley_attributions=shapley, sampling_interval=sampling_interval)

    return shapley, peaks_per_granularity, ci, resource_time_val



