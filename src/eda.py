# Explorative Data Analysis & Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools

_features = 'NetFlow_v3_Features.csv'
feature_names = pd.read_csv(_features)['Feature'].tolist()
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
        df = pd.read_csv(datasets[name], names=col_names)
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

if __name__ == '__main__':
    # visualize_benign_malicious_dist()
    # visualize_attack_types()
    # visualize_histogram()
    visualize_corr_heatmap()

