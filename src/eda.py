# Explorative Data Analysis & Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools

def create_path(filename):
    import os
    datadir = os.path.join(os.path.dirname(__file__), '..', 'data')
    return os.path.join(datadir, filename)

_features = 'NetFlow_v3_Features.csv'
feature_names = pd.read_csv(create_path(_features))['Feature'].tolist()
feature_names = [name.strip() for name in feature_names]
col_names = feature_names + ['Label', 'Attack']

_bot_iot = 'NF-BoT-IoT-v3.csv'
_ids_2018 = 'NF-CICIDS2018-v3.csv'
_unsw_nb15 = 'NF-UNSW-NB15-v3.csv'

datasets = {
    'BoT-IoT': _bot_iot,
    'IDS-2018': _ids_2018,
    'UNSW-NB15': _unsw_nb15
}

def load_dataset(name):
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(datasets.keys())}")
    if type(datasets[name]) is str:
        df = pd.read_csv(create_path(datasets[name]), names=col_names)
        # arr = np.loadtxt(datasets[name], delimiter=',', dtype=str, skiprows=1)
        # df = pd.DataFrame(arr, columns=col_names)
        print(f"Loaded dataset {name} with shape {df.shape}")
        datasets[name] = df
    return datasets[name]

def visualize_benign_malicious_dist():
    '''
    a single bar chart, 3 horizontal groups, each group has 2 bars (blue = benign, orange = malicious)
    '''
    data = []
    for name in datasets:
        df = load_dataset(name)
        benign_count = (df['Label'] == 1).sum()
        total_count = len(df)
        malicious_count = total_count - benign_count
        data.append((name, benign_count, malicious_count))
    df_plot = pd.DataFrame(data, columns=['Dataset', 'Benign', 'Malicious'])
    df_plot.set_index('Dataset', inplace=True)
    df_plot.plot(kind='bar', stacked=False)
    plt.ylabel('Number of Samples')
    plt.title('Benign vs Malicious Sample Distribution')
    plt.xticks(rotation=0)
    plt.savefig('benign_malicious_distribution.png')
    plt.close()
    '''
    write values to csv
    '''
    with open('benign_malicious_distribution.csv', 'w') as f:
        f.write('Dataset,Benign,Malicious,Total\n')
        for index, row in df_plot.iterrows():
            total = row['Benign'] + row['Malicious']
            f.write(f"{index},{row['Benign']},{row['Malicious']},{total}\n")

def visualize_attack_types():
    '''
    3 pie charts, one for each dataset, showing distribution of attack types, remove 'Benign' type
    '''
    for name in datasets:
        df = load_dataset(name)
        attack_counts = df['Attack'].value_counts()
        attack_counts = attack_counts[attack_counts.index != 'Benign']
        with open(f'attack_type_distribution_{name}.csv', 'w') as f:
            f.write('Attack Type,Count\n')
            for attack_type, count in attack_counts.items():
                f.write(f"{attack_type},{count}\n")
        
        total = attack_counts.sum()
        percents = attack_counts / total

        small_types = percents[percents < 0.04].index

        attack_counts_new = attack_counts.copy()
        others_count = attack_counts_new[small_types].sum()
        attack_counts_new = attack_counts_new.drop(small_types)
        attack_counts_new['Others'] = others_count

        plt.figure()
        attack_counts_new.plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title(f'Attack Type Distribution in {name} Dataset')
        plt.ylabel('')
        plt.savefig(f'attack_type_distribution_{name}.png')
        plt.close()

# def find_numeric_columns(df):
#     # numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#     # print(f"{len(numeric_cols)} numeric columns found.")
#     # numeric_cols = []
#     # for col in df.columns:
#     #     if pd.api.types.is_numeric_dtype(df[col]):
#     #         numeric_cols.append(col)
#     # let's say, just not object
#     numeric_cols = [col for col in df.columns if type(df[col].iloc[0]) != str]
#     print(f"{len(numeric_cols)} numeric columns found.")
#     return numeric_cols

# def describe_columns(df):
#     for col in df.columns:
#         print(f"Column: {col}")
#         print(df[col].describe())
#         print()

numeric_cols = [
    # 'L4_DST_PORT',
    'L7_PROTO',
    'IN_BYTES',
    # 'IN_PKTS',
    'OUT_BYTES',
    # 'OUT_PKTS',
    'TCP_FLAGS',
    'CLIENT_TCP_FLAGS',
    'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS',
    'DURATION_IN',
    'DURATION_OUT',
    'MIN_TTL',
    'MAX_TTL',
    'LONGEST_FLOW_PKT',
    'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN',
    'MAX_IP_PKT_LEN',
    'RETRANSMITTED_IN_BYTES',
    # 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES',
    # 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES',
    'TCP_WIN_MAX_IN',
    'TCP_WIN_MAX_OUT',
    'ICMP_IPV4_TYPE',
    'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE',
    'SRC_TO_DST_IAT_MIN',
    'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG',
    'DST_TO_SRC_IAT_MIN',
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    # 'Label'
]

def drop_na_and_non_numeric(col):
    return col.dropna().apply(pd.to_numeric, errors='coerce').dropna()

def visualize_histogram():
    df = load_dataset('UNSW-NB15')
    # numeric_cols = find_numeric_columns(df)
    num_cols = len(numeric_cols)
    cols = 4
    rows = (num_cols + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 4))

    for i, col in enumerate(numeric_cols):
        print(f"Plotting histogram for {col} ({i+1})")
        plt.subplot(rows, cols, i + 1)
        sns.histplot(drop_na_and_non_numeric(df[col]), bins=10, kde=False)
        # plt.xlabel(col)
        plt.xlabel('') # prevent overlap
        # plt.title(f'Histogram of {col}')
        plt.title(col)

    # 避免tight_layout可能冲突
    # plt.tight_layout()   # 可以暂时注释掉
    plt.savefig('numeric_features_histograms.png', bbox_inches='tight')
    plt.close()

def visualize_corr_heatmap():
    df = load_dataset('UNSW-NB15')
    # numeric_cols = find_numeric_columns(df)
    _numeric_cols = numeric_cols.copy()
    _numeric_cols.append('Label')
    df_numeric = df[_numeric_cols].apply(pd.to_numeric, errors='coerce')
    corr_matrix = df_numeric.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig('correlation_heatmap.png', bbox_inches='tight')
    # 保证左边和下边标签不被裁剪
    plt.tight_layout()
    plt.close()

# def visualize_flow_duration_stacked():
#     """
#     (1) Flow Duration Stacked Histogram
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
#     for i, name in enumerate(datasets):
#         print(f"Processing Flow Duration for {name}...")
#         df = load_dataset(name)
        
#         # Calculate Duration in Seconds
#         # Ensure numeric
#         start_ms = pd.to_numeric(df['FLOW_START_MILLISECONDS'], errors='coerce')
#         end_ms = pd.to_numeric(df['FLOW_END_MILLISECONDS'], errors='coerce')
#         duration_sec = (end_ms - start_ms) / 1000.0
#         duration_sec = duration_sec.dropna()
        
#         # Use top 6 frequent attacks to avoid clutter
#         top_attacks = df['Attack'].value_counts().nlargest(7).index.tolist()
#         # remove 'Benign'
#         if 'Benign' in top_attacks:
#             top_attacks.remove('Benign')
        
#         plot_data = []
#         labels = []
#         for atk in top_attacks:
#             # Filter by attack type and corresponding valid duration indices
#             subset = duration_sec[df['Attack'] == atk]
#             if len(subset) > 0:
#                 plot_data.append(subset)
#                 labels.append(atk)
        
#         axes[i].hist(plot_data, bins=30, stacked=True, label=labels, alpha=0.7)
#         axes[i].set_title(f'{name} - Flow Duration')
#         axes[i].set_xlabel('Duration (s)')
#         axes[i].set_ylabel('Frequency')
#         axes[i].legend()
#         # Log scale helps verify short vs long duration visible distribution
#         axes[i].set_yscale('log') 

#     plt.tight_layout()
#     plt.savefig('flow_duration_stacked_hist.png')
#     plt.close()

def visualize_flow_duration_stacked():
    """
    (1) Flow Duration Stacked Histogram
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, name in enumerate(datasets):
        print(f"Processing Flow Duration for {name}...")
        df = load_dataset(name)
        
        # DURATION_IN, DURATION_OUT
        duration_in_ms = pd.to_numeric(df['DURATION_IN'], errors='coerce').dropna()
        duration_out_ms = pd.to_numeric(df['DURATION_OUT'], errors='coerce').dropna()
        # add them
        duration_data = duration_in_ms.add(duration_out_ms, fill_value=0)
        
        top_attacks = df['Attack'].value_counts().nlargest(7).index.tolist()
        if 'Benign' in top_attacks:
            top_attacks.remove('Benign')
        
        plot_data = []
        labels = []
        for atk in top_attacks:
            subset = duration_data[df['Attack'] == atk]
            if len(subset) > 0:
                plot_data.append(subset)
                labels.append(atk)
        
        axes[i].hist(plot_data, bins=30, stacked=True, label=labels, alpha=0.7)
        axes[i].set_title(f'{name} - Flow Duration')
        axes[i].set_xlabel('Duration (ms)')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig('flow_duration_stacked_hist.png')
    plt.close()

def visualize_iat_stacked():
    """
    (2) SRC_TO_DST_IAT_AVG Stacked Histogram
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, name in enumerate(datasets):
        print(f"Processing IAT AVG for {name}...")
        df = load_dataset(name)
        
        iat_col = 'SRC_TO_DST_IAT_AVG'
        iat_data = pd.to_numeric(df[iat_col], errors='coerce').dropna()
        
        top_attacks = df['Attack'].value_counts().nlargest(7).index.tolist()
        if 'Benign' in top_attacks:
            top_attacks.remove('Benign')
        
        plot_data = []
        labels = []
        for atk in top_attacks:
            subset = iat_data[df['Attack'] == atk]
            if len(subset) > 0:
                plot_data.append(subset)
                labels.append(atk)
        
        axes[i].hist(plot_data, bins=30, stacked=True, label=labels, alpha=0.7)
        axes[i].set_title(f'{name} - {iat_col}')
        axes[i].set_xlabel('IAT AVG')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig('iat_avg_stacked_hist.png')
    plt.close()

def visualize_attack_time_series():
    """
    (3) Attack Frequency vs Relative Time over whole duration
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for i, name in enumerate(datasets):
        print(f"Processing Time Series for {name}...")
        df = load_dataset(name)

        # Explicitly convert to numeric and drop invalid/NaN timestamps
        start_ms = pd.to_numeric(df['FLOW_START_MILLISECONDS'], errors='coerce').dropna()
        if start_ms.empty:
            continue

        min_start = start_ms.min()
        max_start = start_ms.max()
        
        duration_minutes = (max_start - min_start) / (1000.0 * 60.0)
        if duration_minutes <= 0:
            print(f"Warning: Dataset {name} has zero or negative duration.")
            continue

        # Calculate relative time in minutes
        # Align Attack column with valid start_ms rows
        rel_time_min = (start_ms - min_start) / (1000.0 * 60.0)
        valid_attacks = df.loc[start_ms.index, 'Attack']

        # Create a temp dataframe
        temp_df = pd.DataFrame({
            'Time_Min': rel_time_min,
            'Attack': valid_attacks
        })

        # Use 100 bins for better resolution
        num_bins = 100
        bin_size = duration_minutes / num_bins
        
        # Calculate integer bin index (0 to num_bins-1)
        temp_df['Bin_Index'] = (temp_df['Time_Min'] / bin_size).astype(int)
        # Clip to ensure max value falls into the last bin
        temp_df['Bin_Index'] = temp_df['Bin_Index'].clip(upper=num_bins - 1)

        # Count attacks in each bin
        # unstack might result in missing bins if no data exists in them
        counts = temp_df.groupby(['Bin_Index', 'Attack']).size().unstack(fill_value=0)

        # Reindex to ensure we have a continuous time axis (0..99), filling gaps with 0
        full_idx = range(num_bins)
        counts = counts.reindex(full_idx, fill_value=0)

        # Map index back to representative time (minutes) for plotting
        counts.index = [x * bin_size for x in counts.index]

        # Select top attacks to plot lines
        top_attacks = counts.sum().nlargest(7).index
        if 'Benign' in top_attacks:
            top_attacks = top_attacks.drop('Benign')
        
        if not top_attacks.empty:
            counts_top = counts[top_attacks]
            counts_top.plot(ax=axes[i], marker='.', markersize=2, linewidth=1)
        
        axes[i].set_title(f'{name} - Attack Freq vs Time')
        axes[i].set_xlabel('Time (minutes from start)')
        axes[i].set_ylabel('Attack Frequency (per bin)')
        axes[i].legend(title='Attack Type')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('attack_freq_timeseries.png')
    plt.close()

if __name__ == '__main__':
    # visualize_benign_malicious_dist()
    # visualize_attack_types()
    # visualize_histogram()
    # visualize_corr_heatmap()
    
    visualize_flow_duration_stacked()
    visualize_iat_stacked()
    visualize_attack_time_series()

