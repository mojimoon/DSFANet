import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from . import config


def make_path(filename):
    datadir = os.path.join(os.path.dirname(__file__), "..", "data")
    return os.path.join(datadir, filename)


class IDSDataset(Dataset):
    def __init__(self, x_static, x_temporal, y):
        self.x_static = np.ascontiguousarray(np.asarray(x_static, dtype=np.float32))
        self.x_temporal = np.ascontiguousarray(np.asarray(x_temporal, dtype=np.float32))
        self.y = np.ascontiguousarray(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_static[idx]),
            torch.from_numpy(self.x_temporal[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class CombinedFeatureDataset(Dataset):
    def __init__(self, x_static, x_temporal, y, t_stream_dim: int | None = None):
        self.x_static = np.ascontiguousarray(np.asarray(x_static, dtype=np.float32))
        self.x_temporal = np.ascontiguousarray(np.asarray(x_temporal, dtype=np.float32))
        self.y = np.ascontiguousarray(np.asarray(y, dtype=np.int64))
        self.t_stream_dim = t_stream_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_static = self.x_static[idx]
        x_temporal = self.x_temporal[idx]
        if self.t_stream_dim is None:
            x_temporal_use = x_temporal
        else:
            x_temporal_use = x_temporal[: self.t_stream_dim]
        x_combined = np.concatenate([x_static, x_temporal_use], axis=0).astype(np.float32, copy=False)
        return (
            torch.from_numpy(x_static),
            torch.from_numpy(x_combined),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = make_path(filepath)
        self.scaler_static = MinMaxScaler()
        self.scaler_temporal = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.used_static_cols: list[str] = []
        self.used_temporal_cols: list[str] = []
        self.used_t_stream_cols: list[str] = []
        self.used_timestamp_cols: list[str] = []
        self.used_temporal_all_cols: list[str] = []
        self.log_scale_cols: list[str] = []

    @staticmethod
    def _should_log_scale(column_name: str) -> bool:
        return False

        """
        name = column_name.upper().strip()
        if name in {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"}:
            return False
        keywords = [
            "BYTES",
            "PKTS",
            "DURATION",
            "IAT",
            "THROUGHPUT",
            "RETRANSMITTED",
            "WIN_MAX",
            "TTL",
            "FLOW_PKT",
            "IP_PKT_LEN",
        ]
        return any(k in name for k in keywords)
        """

    def apply_log_scale(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        applied: list[str] = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            if not self._should_log_scale(col):
                continue
            col_values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if (col_values < 0).any():
                continue
            df[col] = np.log1p(col_values)
            applied.append(col)
        self.log_scale_cols = sorted(set(applied))
        return df

    def clean_data(self, df):
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        cols_to_drop = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=existing_cols, inplace=True)
        return df

    def prepare_data(self):
        print(f"Loading data from {self.filepath}...")
        df = pd.read_csv(self.filepath, low_memory=False)

        if config.LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{config.LABEL_COLUMN}' not found in dataset.")

        df = self.clean_data(df)

        static_cols = [c for c in config.STATIC_FEATURES if c in df.columns]
        t_stream_cols = [c for c in config.T_STREAM_FEATURES if c in df.columns]
        timestamp_cols = [c for c in config.TIMESTAMP_FEATURES if c in df.columns]
        temporal_cols = t_stream_cols + timestamp_cols
        self.used_static_cols = static_cols
        self.used_t_stream_cols = t_stream_cols
        self.used_timestamp_cols = timestamp_cols
        self.used_temporal_all_cols = temporal_cols
        self.used_temporal_cols = temporal_cols
        feature_cols = static_cols + temporal_cols
        df = self.apply_log_scale(df, feature_cols)

        print(f"Features mapped: {len(static_cols)} Static, {len(t_stream_cols)} T-stream, {len(timestamp_cols)} Timestamps")

        if config.TEST_MODE:
            target_n = min(int(config.TEST_SIZE), len(df))
            if target_n < len(df):
                y_for_sample = df[config.LABEL_COLUMN]
                unique_classes = pd.Series(y_for_sample).nunique(dropna=False)
                if unique_classes > 1:
                    try:
                        df, _ = train_test_split(
                            df,
                            train_size=target_n,
                            random_state=42,
                            stratify=y_for_sample,
                        )
                        print(f"Test mode: using stratified {target_n} samples.")
                    except Exception:
                        df = df.iloc[:target_n]
                        print(f"Test mode: fallback to first {target_n} samples.")
                else:
                    df = df.iloc[:target_n]
                    print(f"Test mode: single-class source, using first {target_n} samples.")
            else:
                print(f"Test mode: using all {target_n} samples.")

        x_static = (
            df[static_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        x_temporal = (
            df[temporal_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )

        y_raw = df[config.LABEL_COLUMN]
        if y_raw.dtype == "object":
            y = self.label_encoder.fit_transform(y_raw.astype(str))
            print(f"Classes encoded: {self.label_encoder.classes_}")
        else:
            y = y_raw.values.astype(np.int64)

        x_static_train, x_static_test, x_temporal_train, x_temporal_test, y_train, y_test = train_test_split(
            x_static,
            x_temporal,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        x_static_train = self.scaler_static.fit_transform(x_static_train)
        x_static_test = self.scaler_static.transform(x_static_test)

        x_temporal_train = self.scaler_temporal.fit_transform(x_temporal_train)
        x_temporal_test = self.scaler_temporal.transform(x_temporal_test)

        print(
            f"Data prepared. Static shape: {x_static_train.shape}, Temporal shape: {x_temporal_train.shape}"
        )

        return (x_static_train, x_temporal_train, y_train), (x_static_test, x_temporal_test, y_test)


def get_dataloaders(train_data, test_data, batch_size):
    train_dataset = IDSDataset(*train_data)
    test_dataset = IDSDataset(*test_data)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader


def get_combined_dataloaders(train_data, test_data, batch_size, t_stream_dim: int | None = None):
    train_dataset = CombinedFeatureDataset(*train_data, t_stream_dim=t_stream_dim)
    test_dataset = CombinedFeatureDataset(*test_data, t_stream_dim=t_stream_dim)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader


def extract_benign_samples(filepath: str, max_samples: int | None = None):
    preprocessor = DataPreprocessor(filepath)
    print(f"Loading benign samples from {preprocessor.filepath}...")
    df = pd.read_csv(preprocessor.filepath, low_memory=False)

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{config.LABEL_COLUMN}' not found in dataset.")

    df = preprocessor.clean_data(df)
    static_cols = [c for c in config.STATIC_FEATURES if c in df.columns]
    temporal_cols = [c for c in (config.T_STREAM_FEATURES + config.TIMESTAMP_FEATURES) if c in df.columns]
    df = preprocessor.apply_log_scale(df, static_cols + temporal_cols)

    x_static = (
        df[static_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float32)
    )
    x_temporal = (
        df[temporal_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float32)
    )

    y_raw = df[config.LABEL_COLUMN]
    if y_raw.dtype == "object":
        y = preprocessor.label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values.astype(np.int64)

    x_static_scaled = preprocessor.scaler_static.fit_transform(x_static)
    x_temporal_scaled = preprocessor.scaler_temporal.fit_transform(x_temporal)

    benign_idx = np.where(y == 0)[0]
    if len(benign_idx) == 0:
        raise ValueError("No benign samples (Label=0) found in dataset.")

    if max_samples is not None and max_samples > 0 and len(benign_idx) > max_samples:
        benign_idx = np.random.choice(benign_idx, size=max_samples, replace=False)

    return x_static_scaled[benign_idx], x_temporal_scaled[benign_idx]