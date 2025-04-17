import emb_shapley_lib as eshap
import pandas as pd
import pickle
minute = 60
ten_minute = 10 * minute
hour = minute * 60
day = 24 * hour
week = day * 7

def emb_shapley_azure(df_azure, node='clr', resource='cpu', time_granularities = [3 * day, 8 * hour, hour, 5 * minute]):
    lifetime = 4 * 365 / 30 # 1 month
    if node == 'clr':
        imec_cpu_chip_cf = 20540 # gCO2eq
        ACT_cpu_chip_cf = 18530 # gCO2eq
        num_cores_per_cpu = 48
        num_cpus_per_node = 2
        cooling_cf = 24210 # gCO2eq
        gb_per_node = 192
        dram_cf = 146875 # gCO2eq 
        ssd_cf_per_gb =  160 # gCO2eq, from Dirty Secrets of SSDs paper
        ssd_cap_per_node = 480 # GB
        ssd_cf = ssd_cf_per_gb * ssd_cap_per_node # gCO2eq
        mb_cf = 109670 # gCO2eq
        chassis_cf = 34300 # gCO2eq
        peripheral_cf = 59170 # gCO2eq
        psu_cf = 30016 # gCO2eq
    elif node == 'storage':
        imec_cpu_chip_cf = 6740 # gCO2eq
        ACT_cpu_chip_cf = 9670 # gCO2eq
        num_cores_per_cpu = 20
        num_cpus_per_node = 2
        cooling_cf = 15340 # gCO2eq
        gb_per_node = 64
        dram_cf = 74434 # gCO2eq 
        ssd_cf_per_gb =  160 # gCO2eq, from Dirty Secrets of SSDs paper
        ssd_cap_per_node = 480 # GB
        ssd_cf = ssd_cf_per_gb * ssd_cap_per_node # gCO2eq
        mb_cf = 106129 # gCO2eq
        chassis_cf = 34300 # gCO2eq
        peripheral_cf = 59170 # gCO2eq
        psu_cf = 30016 # gCO2eq

    cpu_chip_cf_per_cpu = imec_cpu_chip_cf
    cpu_cf = cpu_chip_cf_per_cpu * num_cpus_per_node # gCO2eq
    cpu_cf_per_core = cpu_cf / num_cores_per_cpu # gCO2eq
    
    cpu_and_cooling_cf = cpu_cf + cooling_cf # gCO2eq
    cpu_and_cooling_cf_per_core = cpu_and_cooling_cf / num_cores_per_cpu # gCO2eq

    dram_cf_per_gb =  dram_cf / gb_per_node# gCO2eq

    other_cf = mb_cf + chassis_cf + peripheral_cf + psu_cf + ssd_cf # gCO2eq
    node_cf = (cpu_and_cooling_cf + dram_cf + ssd_cf + mb_cf + chassis_cf + peripheral_cf + psu_cf)

    kWh_to_J = 3.6e6 # J/kWh
    mem_ci = (dram_cf + 0.5 * other_cf) / gb_per_node / lifetime # gCO2eq
    cpu_ci = (cpu_and_cooling_cf + 0.5 * other_cf) / (num_cores_per_cpu * num_cpus_per_node) / lifetime # gCO2eq
    node_ci = node_cf / lifetime # gCO2eq

    sampling_interval = 300
    offset = 0

    if resource == 'cpu':
        cores_available = df_azure['cpu allocated'].max()
        total_cpu_cf = cpu_ci * cores_available
        cpu_shapley, cpu_peaks, cpu_ci, cpu_resource_time = eshap.shapley_attribution(df_azure, 'timestamp', 'cpu allocated', time_granularities, total_cpu_cf, sampling_interval, offset)
        cpu_ci_3_day = cpu_ci[0]
        cpu_ci_8_hour = cpu_ci[1]
        cpu_ci_hourly = cpu_ci[len(time_granularities) - 2]
        cpu_ci_5_min = cpu_ci[len(time_granularities) - 1]
        # cpu_ci_3_day = pd.DataFrame(cpu_ci_3_day)
        # cpu_ci_8_hour = pd.DataFrame(cpu_ci_8_hour)
        cpu_ci_hourly = pd.DataFrame(cpu_ci_hourly, columns=['embodied ci (gCO2eq/core-second)'])
        cpu_ci_hourly['timestamp'] = cpu_ci_hourly.index * hour
        cpu_ci_5_min = pd.DataFrame(cpu_ci_5_min, columns=['embodied ci (gCO2eq/core-second)'])
        cpu_ci_5_min['timestamp'] = df_azure['timestamp']
        return cpu_ci_5_min, cpu_ci_hourly
    elif resource == 'mem':
        memory_available = df_azure['mem allocated (gb)'].max()
        total_mem_cf = mem_ci * memory_available
        mem_shapley, mem_peaks, mem_ci, mem_resource_time = eshap.shapley_attribution(df_azure, 'timestamp', 'mem allocated (gb)', time_granularities, total_mem_cf, sampling_interval, offset)
        mem_ci_3_day = mem_ci[0]
        mem_ci_8_hour = mem_ci[1]
        mem_ci_hourly = mem_ci[len(time_granularities) - 2]
        mem_ci_5_min = mem_ci[len(time_granularities) - 1]
        # mem_ci_3_day = pd.DataFrame(mem_ci_3_day)
        # mem_ci_8_hour = pd.DataFrame(mem_ci_8_hour)
        mem_ci_hourly = pd.DataFrame(mem_ci_hourly, columns=['embodied ci (gCO2eq/core-second)'])
        mem_ci_hourly['timestamp'] = mem_ci_hourly.index * hour
        mem_ci_5_min = pd.DataFrame(mem_ci_5_min, columns=['embodied ci (gCO2eq/core-second)'])
        mem_ci_5_min['timestamp'] = df_azure['timestamp']
        return mem_ci_5_min, mem_ci_hourly