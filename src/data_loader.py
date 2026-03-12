import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def _largest_remainder_allocation(counts: dict[int, int], target_total: int) -> dict[int, int]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0 for k in counts}
    if target_total <= 0:
        return {k: 0 for k in counts}
    if target_total >= total:
        return dict(counts)

    exact = {k: counts[k] * target_total / total for k in counts}
    alloc = {k: int(np.floor(v)) for k, v in exact.items()}

    positive_classes = [k for k, v in counts.items() if v > 0]
    if target_total >= len(positive_classes):
        for key in positive_classes:
            if alloc[key] == 0:
                alloc[key] = 1

    assigned = sum(alloc.values())
    if assigned > target_total:
        over = assigned - target_total
        for key in sorted(alloc, key=lambda item: (alloc[item], counts[item]), reverse=True):
            if over == 0:
                break
            min_allowed = 1 if key in positive_classes and target_total >= len(positive_classes) else 0
            if alloc[key] > min_allowed:
                alloc[key] -= 1
                over -= 1

    assigned = sum(alloc.values())
    remaining = target_total - assigned
    remainders = sorted(exact.keys(), key=lambda item: (exact[item] - np.floor(exact[item]), counts[item]), reverse=True)
    for key in remainders:
        if remaining == 0:
            break
        if alloc[key] < counts[key]:
            alloc[key] += 1
            remaining -= 1

    return alloc


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
        self.chunk_size = int(getattr(config, "CSV_CHUNK_SIZE", 200000))
        self.scale_chunk_size = int(getattr(config, "SCALE_CHUNK_SIZE", 500000))
        self._artifact_dir = Path(tempfile.mkdtemp(prefix=f"fyp_preproc_{Path(self.filepath).stem}_"))

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
        self.log_scale_cols = sorted(set(self.log_scale_cols).union(applied))
        return df

    def clean_data(self, df):
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        cols_to_drop = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)
        return df

    def _feature_columns(self):
        header_cols = pd.read_csv(self.filepath, nrows=0).columns.tolist()
        static_cols = [c for c in config.STATIC_FEATURES if c in header_cols]
        t_stream_cols = [c for c in config.T_STREAM_FEATURES if c in header_cols]
        timestamp_cols = [c for c in config.TIMESTAMP_FEATURES if c in header_cols]
        temporal_cols = t_stream_cols + timestamp_cols

        self.used_static_cols = static_cols
        self.used_t_stream_cols = t_stream_cols
        self.used_timestamp_cols = timestamp_cols
        self.used_temporal_all_cols = temporal_cols
        self.used_temporal_cols = temporal_cols
        return static_cols, t_stream_cols, timestamp_cols, temporal_cols

    def _iter_csv_chunks(self, usecols: list[str]):
        return pd.read_csv(
            self.filepath,
            usecols=usecols,
            low_memory=False,
            chunksize=self.chunk_size,
        )

    def _fit_label_encoder(self) -> tuple[dict[str, int], dict[int, int], int]:
        counts_raw: dict[str, int] = {}
        total_rows = 0
        for chunk in self._iter_csv_chunks([config.LABEL_COLUMN]):
            labels = chunk[config.LABEL_COLUMN].fillna(0).astype(str)
            values, counts = np.unique(labels.to_numpy(), return_counts=True)
            total_rows += int(len(labels))
            for value, count in zip(values.tolist(), counts.tolist()):
                counts_raw[str(value)] = counts_raw.get(str(value), 0) + int(count)

        classes = sorted(counts_raw.keys())
        self.label_encoder.fit(classes)
        label_to_int = {label: int(code) for label, code in zip(classes, self.label_encoder.transform(classes))}
        class_counts = {label_to_int[label]: count for label, count in counts_raw.items()}
        print(f"Classes encoded: {self.label_encoder.classes_}")
        return label_to_int, class_counts, total_rows

    def _sample_and_split_counts(self, class_counts: dict[int, int], total_rows: int) -> tuple[dict[int, int], dict[int, int]]:
        if config.TEST_MODE:
            sample_total = min(int(config.TEST_SIZE), total_rows)
            print(f"Test mode enabled: target {sample_total} samples from source size {total_rows}.")
        else:
            sample_total = total_rows

        sample_counts = _largest_remainder_allocation(class_counts, sample_total)
        train_counts: dict[int, int] = {}
        for label, count in sample_counts.items():
            if count <= 1:
                train_counts[label] = count
                continue
            test_count = int(round(count * 0.2))
            test_count = max(1, min(test_count, count - 1))
            train_counts[label] = count - test_count
        return sample_counts, train_counts

    def _open_memmap(self, name: str, shape: tuple[int, ...], dtype):
        path = self._artifact_dir / f"{name}.npy"
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    @staticmethod
    def _set_scaler_state(scaler: MinMaxScaler, data_min: np.ndarray, data_max: np.ndarray, n_samples: int) -> None:
        data_min = np.asarray(data_min, dtype=np.float64)
        data_max = np.asarray(data_max, dtype=np.float64)
        data_range = data_max - data_min
        safe_range = np.where(data_range == 0, 1.0, data_range)
        scaler.data_min_ = data_min
        scaler.data_max_ = data_max
        scaler.data_range_ = data_range
        scaler.scale_ = 1.0 / safe_range
        scaler.min_ = -data_min * scaler.scale_
        scaler.n_features_in_ = data_min.shape[0]
        scaler.n_samples_seen_ = int(n_samples)

    def _fit_minmax_stats(self, array: np.memmap) -> tuple[np.ndarray, np.ndarray]:
        n_rows, n_cols = array.shape
        mins = np.full(n_cols, np.inf, dtype=np.float64)
        maxs = np.full(n_cols, -np.inf, dtype=np.float64)
        for start in range(0, n_rows, self.scale_chunk_size):
            end = min(start + self.scale_chunk_size, n_rows)
            block = np.asarray(array[start:end], dtype=np.float32)
            if block.size == 0:
                continue
            mins = np.minimum(mins, block.min(axis=0))
            maxs = np.maximum(maxs, block.max(axis=0))
        mins[~np.isfinite(mins)] = 0.0
        maxs[~np.isfinite(maxs)] = 0.0
        return mins.astype(np.float32), maxs.astype(np.float32)

    def _transform_minmax_inplace(self, array: np.memmap, mins: np.ndarray, maxs: np.ndarray) -> None:
        denom = maxs - mins
        denom = np.where(denom == 0, 1.0, denom).astype(np.float32)
        mins = mins.astype(np.float32)
        n_rows = array.shape[0]
        for start in range(0, n_rows, self.scale_chunk_size):
            end = min(start + self.scale_chunk_size, n_rows)
            block = np.asarray(array[start:end], dtype=np.float32)
            if block.size == 0:
                continue
            block = (block - mins) / denom
            array[start:end] = block.astype(np.float32, copy=False)
        array.flush()

    def _write_split_memmaps(
        self,
        static_cols: list[str],
        temporal_cols: list[str],
        label_to_int: dict[str, int],
        class_counts: dict[int, int],
        sample_counts: dict[int, int],
        train_counts: dict[int, int],
    ):
        train_rows = int(sum(train_counts.values()))
        total_rows = int(sum(sample_counts.values()))
        test_rows = total_rows - train_rows

        x_static_train = self._open_memmap("x_static_train", (train_rows, len(static_cols)), np.float32)
        x_temporal_train = self._open_memmap("x_temporal_train", (train_rows, len(temporal_cols)), np.float32)
        y_train = self._open_memmap("y_train", (train_rows,), np.int64)

        x_static_test = self._open_memmap("x_static_test", (test_rows, len(static_cols)), np.float32)
        x_temporal_test = self._open_memmap("x_temporal_test", (test_rows, len(temporal_cols)), np.float32)
        y_test = self._open_memmap("y_test", (test_rows,), np.int64)

        full_remaining_total = dict(class_counts)
        remaining_sample = dict(sample_counts)
        remaining_train = dict(train_counts)
        rng = np.random.default_rng(42)
        train_pos = 0
        test_pos = 0

        usecols = static_cols + temporal_cols + [config.LABEL_COLUMN]
        for chunk in self._iter_csv_chunks(usecols):
            chunk = self.clean_data(chunk)
            chunk = self.apply_log_scale(chunk, static_cols + temporal_cols)

            static_block = chunk.reindex(columns=static_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
            temporal_block = chunk.reindex(columns=temporal_cols, fill_value=0).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
            labels_block = chunk[config.LABEL_COLUMN].fillna(0).astype(str).map(label_to_int).to_numpy(dtype=np.int64, copy=False)

            for label in np.unique(labels_block):
                label = int(label)
                class_idx = np.flatnonzero(labels_block == label)
                chunk_count = int(len(class_idx))
                if chunk_count == 0:
                    continue

                rem_total = full_remaining_total.get(label, 0)
                rem_sample = remaining_sample.get(label, 0)
                rem_train = remaining_train.get(label, 0)

                if rem_total <= 0:
                    continue

                if rem_sample <= 0:
                    full_remaining_total[label] = rem_total - chunk_count
                    continue

                if rem_sample >= rem_total:
                    sample_in_chunk = chunk_count
                else:
                    sample_in_chunk = int(
                        rng.hypergeometric(
                            ngood=chunk_count,
                            nbad=rem_total - chunk_count,
                            nsample=rem_sample,
                        )
                    )

                selected_idx = np.array([], dtype=np.int64)
                train_idx = np.array([], dtype=np.int64)
                test_idx = np.array([], dtype=np.int64)
                if sample_in_chunk > 0:
                    selected_idx = rng.choice(class_idx, size=sample_in_chunk, replace=False)
                    if rem_train <= 0:
                        train_in_chunk = 0
                    elif rem_train >= rem_sample:
                        train_in_chunk = sample_in_chunk
                    else:
                        train_in_chunk = int(
                            rng.hypergeometric(
                                ngood=sample_in_chunk,
                                nbad=rem_sample - sample_in_chunk,
                                nsample=rem_train,
                            )
                        )

                    if train_in_chunk > 0:
                        train_idx = rng.choice(selected_idx, size=train_in_chunk, replace=False)
                        train_mask = np.zeros(sample_in_chunk, dtype=bool)
                        train_mask[np.isin(selected_idx, train_idx)] = True
                        test_idx = selected_idx[~train_mask]
                    else:
                        test_idx = selected_idx
                else:
                    train_in_chunk = 0

                if train_idx.size > 0:
                    rows = np.sort(train_idx)
                    count = int(len(rows))
                    x_static_train[train_pos : train_pos + count] = static_block[rows]
                    x_temporal_train[train_pos : train_pos + count] = temporal_block[rows]
                    y_train[train_pos : train_pos + count] = label
                    train_pos += count

                if test_idx.size > 0:
                    rows = np.sort(test_idx)
                    count = int(len(rows))
                    x_static_test[test_pos : test_pos + count] = static_block[rows]
                    x_temporal_test[test_pos : test_pos + count] = temporal_block[rows]
                    y_test[test_pos : test_pos + count] = label
                    test_pos += count

                full_remaining_total[label] = rem_total - chunk_count
                remaining_sample[label] = rem_sample - sample_in_chunk
                remaining_train[label] = rem_train - train_in_chunk

        x_static_train.flush()
        x_temporal_train.flush()
        y_train.flush()
        x_static_test.flush()
        x_temporal_test.flush()
        y_test.flush()
        return (x_static_train, x_temporal_train, y_train), (x_static_test, x_temporal_test, y_test)

    def prepare_data(self):
        print(f"Loading data from {self.filepath}...")
        static_cols, t_stream_cols, timestamp_cols, temporal_cols = self._feature_columns()
        if config.LABEL_COLUMN not in pd.read_csv(self.filepath, nrows=0).columns:
            raise ValueError(f"Label column '{config.LABEL_COLUMN}' not found in dataset.")

        print(f"Features mapped: {len(static_cols)} Static, {len(t_stream_cols)} T-stream, {len(timestamp_cols)} Timestamps")
        label_to_int, class_counts, total_rows = self._fit_label_encoder()
        sample_counts, train_counts = self._sample_and_split_counts(class_counts, total_rows)
        sampled_rows = int(sum(sample_counts.values()))
        train_rows = int(sum(train_counts.values()))
        test_rows = sampled_rows - train_rows
        if config.TEST_MODE:
            print(f"Test mode: using stratified {sampled_rows} samples.")

        train_data, test_data = self._write_split_memmaps(
            static_cols=static_cols,
            temporal_cols=temporal_cols,
            label_to_int=label_to_int,
            class_counts=class_counts,
            sample_counts=sample_counts,
            train_counts=train_counts,
        )

        x_static_train, x_temporal_train, y_train = train_data
        x_static_test, x_temporal_test, y_test = test_data

        static_min, static_max = self._fit_minmax_stats(x_static_train)
        temporal_min, temporal_max = self._fit_minmax_stats(x_temporal_train)
        self._set_scaler_state(self.scaler_static, static_min, static_max, train_rows)
        self._set_scaler_state(self.scaler_temporal, temporal_min, temporal_max, train_rows)

        self._transform_minmax_inplace(x_static_train, static_min, static_max)
        self._transform_minmax_inplace(x_static_test, static_min, static_max)
        self._transform_minmax_inplace(x_temporal_train, temporal_min, temporal_max)
        self._transform_minmax_inplace(x_temporal_test, temporal_min, temporal_max)

        print(f"Data prepared. Static shape: {x_static_train.shape}, Temporal shape: {x_temporal_train.shape}")
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
    train_data, test_data = preprocessor.prepare_data()
    x_static_train, x_temporal_train, y_train = train_data
    x_static_test, x_temporal_test, y_test = test_data

    benign_train_idx = np.flatnonzero(np.asarray(y_train) == 0)
    benign_test_idx = np.flatnonzero(np.asarray(y_test) == 0)
    total_benign = int(len(benign_train_idx) + len(benign_test_idx))
    if total_benign == 0:
        raise ValueError("No benign samples (Label=0) found in dataset.")

    if max_samples is None or max_samples <= 0 or total_benign <= max_samples:
        keep_train = benign_train_idx
        keep_test = benign_test_idx
    else:
        rng = np.random.default_rng(42)
        train_quota = int(round(max_samples * len(benign_train_idx) / total_benign))
        train_quota = min(train_quota, len(benign_train_idx), max_samples)
        test_quota = max_samples - train_quota
        test_quota = min(test_quota, len(benign_test_idx))
        if train_quota + test_quota < max_samples:
            spare = max_samples - (train_quota + test_quota)
            train_extra = min(spare, len(benign_train_idx) - train_quota)
            train_quota += train_extra
            spare -= train_extra
            test_quota += min(spare, len(benign_test_idx) - test_quota)

        keep_train = rng.choice(benign_train_idx, size=train_quota, replace=False) if train_quota > 0 else np.array([], dtype=int)
        keep_test = rng.choice(benign_test_idx, size=test_quota, replace=False) if test_quota > 0 else np.array([], dtype=int)

    benign_static = []
    benign_temporal = []
    if len(keep_train) > 0:
        benign_static.append(np.asarray(x_static_train[keep_train], dtype=np.float32))
        benign_temporal.append(np.asarray(x_temporal_train[keep_train], dtype=np.float32))
    if len(keep_test) > 0:
        benign_static.append(np.asarray(x_static_test[keep_test], dtype=np.float32))
        benign_temporal.append(np.asarray(x_temporal_test[keep_test], dtype=np.float32))

    return np.concatenate(benign_static, axis=0), np.concatenate(benign_temporal, axis=0)
