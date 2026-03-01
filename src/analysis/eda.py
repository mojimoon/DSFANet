import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    return os.path.abspath(os.path.join(data_dir, filename))


_features = "NetFlow_v3_Features.csv"
feature_names = pd.read_csv(make_path(_features))["Feature"].tolist()
feature_names = [name.strip() for name in feature_names]
col_names = feature_names + ["Label", "Attack"]

datasets = {
    "BoT-IoT": "NF-BoT-IoT-v3.csv",
    "IDS-2018": "NF-CICIDS2018-v3.csv",
    "UNSW-NB15": "NF-UNSW-NB15-v3.csv",
}


def load_dataset(name):
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(datasets.keys())}")
    if isinstance(datasets[name], str):
        df = pd.read_csv(make_path(datasets[name]), names=col_names)
        print(f"Loaded dataset {name} with shape {df.shape}")
        datasets[name] = df
    return datasets[name]


def visualize_benign_malicious_dist():
    """Generate grouped bar chart for benign/malicious counts."""
    data = []
    for name in datasets:
        df = load_dataset(name)
        benign_count = (df["Label"] == 1).sum()
        total_count = len(df)
        malicious_count = total_count - benign_count
        data.append((name, benign_count, malicious_count))

    df_plot = pd.DataFrame(data, columns=["Dataset", "Benign", "Malicious"]).set_index("Dataset")
    df_plot.plot(kind="bar", stacked=False)
    plt.ylabel("Number of Samples")
    plt.title("Benign vs Malicious Sample Distribution")
    plt.xticks(rotation=0)
    plt.savefig("benign_malicious_distribution.png")
    plt.close()

    with open("benign_malicious_distribution.csv", "w", encoding="utf-8") as output:
        output.write("Dataset,Benign,Malicious,Total\n")
        for index, row in df_plot.iterrows():
            total = row["Benign"] + row["Malicious"]
            output.write(f"{index},{row['Benign']},{row['Malicious']},{total}\n")


def visualize_attack_types():
    """Generate pie charts for attack-type distribution excluding Benign."""
    for name in datasets:
        df = load_dataset(name)
        attack_counts = df["Attack"].value_counts()
        attack_counts = attack_counts[attack_counts.index != "Benign"]

        with open(f"attack_type_distribution_{name}.csv", "w", encoding="utf-8") as output:
            output.write("Attack Type,Count\n")
            for attack_type, count in attack_counts.items():
                output.write(f"{attack_type},{count}\n")

        total = attack_counts.sum()
        percents = attack_counts / total
        small_types = percents[percents < 0.04].index

        attack_counts_new = attack_counts.copy()
        others_count = attack_counts_new[small_types].sum()
        attack_counts_new = attack_counts_new.drop(small_types)
        attack_counts_new["Others"] = others_count

        plt.figure()
        attack_counts_new.plot.pie(autopct="%1.1f%%", startangle=90)
        plt.title(f"Attack Type Distribution in {name} Dataset")
        plt.ylabel("")
        plt.savefig(f"attack_type_distribution_{name}.png")
        plt.close()


numeric_cols = [
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_OUT_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
]


def drop_na_and_non_numeric(column):
    return column.dropna().apply(pd.to_numeric, errors="coerce").dropna()


def visualize_histogram():
    df = load_dataset("UNSW-NB15")
    num_cols = len(numeric_cols)
    cols = 4
    rows = (num_cols + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 4))
    for idx, column in enumerate(numeric_cols):
        print(f"Plotting histogram for {column} ({idx + 1})")
        plt.subplot(rows, cols, idx + 1)
        sns.histplot(drop_na_and_non_numeric(df[column]), bins=10, kde=False)
        plt.xlabel("")
        plt.title(column)

    plt.savefig("numeric_features_histograms.png", bbox_inches="tight")
    plt.close()


def visualize_corr_heatmap():
    df = load_dataset("UNSW-NB15")
    selected_cols = numeric_cols.copy() + ["Label"]
    df_numeric = df[selected_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig("correlation_heatmap.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close()


def visualize_flow_duration_stacked():
    """Plot stacked histograms for attack-wise flow duration."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for idx, name in enumerate(datasets):
        print(f"Processing flow duration for {name}...")
        df = load_dataset(name)

        duration_in_ms = pd.to_numeric(df["DURATION_IN"], errors="coerce").dropna()
        duration_out_ms = pd.to_numeric(df["DURATION_OUT"], errors="coerce").dropna()
        duration_data = duration_in_ms.add(duration_out_ms, fill_value=0)

        top_attacks = df["Attack"].value_counts().nlargest(7).index.tolist()
        if "Benign" in top_attacks:
            top_attacks.remove("Benign")

        plot_data = []
        labels = []
        for attack in top_attacks:
            subset = duration_data[df["Attack"] == attack]
            if len(subset) > 0:
                plot_data.append(subset)
                labels.append(attack)

        axes[idx].hist(plot_data, bins=30, stacked=True, label=labels, alpha=0.7)
        axes[idx].set_title(f"{name} - Flow Duration")
        axes[idx].set_xlabel("Duration (ms)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend()
        axes[idx].set_yscale("log")

    plt.tight_layout()
    plt.savefig("flow_duration_stacked_hist.png")
    plt.close()


def visualize_iat_stacked():
    """Plot stacked histograms for attack-wise SRC_TO_DST_IAT_AVG."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for idx, name in enumerate(datasets):
        print(f"Processing IAT AVG for {name}...")
        df = load_dataset(name)

        iat_data = pd.to_numeric(df["SRC_TO_DST_IAT_AVG"], errors="coerce").dropna()
        top_attacks = df["Attack"].value_counts().nlargest(7).index.tolist()
        if "Benign" in top_attacks:
            top_attacks.remove("Benign")

        plot_data = []
        labels = []
        for attack in top_attacks:
            subset = iat_data[df["Attack"] == attack]
            if len(subset) > 0:
                plot_data.append(subset)
                labels.append(attack)

        axes[idx].hist(plot_data, bins=30, stacked=True, label=labels, alpha=0.7)
        axes[idx].set_title(f"{name} - SRC_TO_DST_IAT_AVG")
        axes[idx].set_xlabel("IAT AVG")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend()
        axes[idx].set_yscale("log")

    plt.tight_layout()
    plt.savefig("iat_avg_stacked_hist.png")
    plt.close()


def visualize_attack_time_series():
    """Plot attack frequency over relative timeline for each dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for idx, name in enumerate(datasets):
        print(f"Processing time series for {name}...")
        df = load_dataset(name)

        start_ms = pd.to_numeric(df["FLOW_START_MILLISECONDS"], errors="coerce").dropna()
        if start_ms.empty:
            continue

        min_start = start_ms.min()
        max_start = start_ms.max()
        duration_minutes = (max_start - min_start) / (1000.0 * 60.0)
        if duration_minutes <= 0:
            print(f"Warning: dataset {name} has non-positive duration.")
            continue

        rel_time_min = (start_ms - min_start) / (1000.0 * 60.0)
        valid_attacks = df.loc[start_ms.index, "Attack"]
        temp_df = pd.DataFrame({"Time_Min": rel_time_min, "Attack": valid_attacks})

        num_bins = 100
        bin_size = duration_minutes / num_bins
        temp_df["Bin_Index"] = (temp_df["Time_Min"] / bin_size).astype(int)
        temp_df["Bin_Index"] = temp_df["Bin_Index"].clip(upper=num_bins - 1)

        counts = temp_df.groupby(["Bin_Index", "Attack"]).size().unstack(fill_value=0)
        counts = counts.reindex(range(num_bins), fill_value=0)
        counts.index = [bin_index * bin_size for bin_index in counts.index]

        top_attacks = counts.sum().nlargest(7).index
        if "Benign" in top_attacks:
            top_attacks = top_attacks.drop("Benign")

        if not top_attacks.empty:
            counts[top_attacks].plot(ax=axes[idx], marker=".", markersize=2, linewidth=1)

        axes[idx].set_title(f"{name} - Attack Freq vs Time")
        axes[idx].set_xlabel("Time (minutes from start)")
        axes[idx].set_ylabel("Attack Frequency (per bin)")
        axes[idx].legend(title="Attack Type")
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("attack_freq_timeseries.png")
    plt.close()


if __name__ == "__main__":
    visualize_flow_duration_stacked()
    visualize_iat_stacked()
    visualize_attack_time_series()
