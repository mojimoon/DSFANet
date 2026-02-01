import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import config

def make_path(filename):
    import os
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(datadir, filename)

class IDSDataset(Dataset):
    def __init__(self, x_static, x_temporal, y):
        self.x_static = torch.FloatTensor(x_static)
        self.x_temporal = torch.FloatTensor(x_temporal)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_static[idx], self.x_temporal[idx], self.y[idx]

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = make_path(filepath)
        self.scaler_static = MinMaxScaler()
        self.scaler_temporal = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def clean_data(self, df):
        """数据清洗：处理缺失值，转换非数值类型"""
        df = df.copy()
        
        # 1. 替换无穷大值为 NaN，然后填充
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # 2. 删除高基数标识符列 (IP地址)，防止过拟合
        cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'] 
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=existing_cols, inplace=True)
        
        return df

    def prepare_data(self):
        print(f"Loading data from {self.filepath}...")
        # 低内存模式 false 以防止混合类型推断错误
        df = pd.read_csv(self.filepath, low_memory=False)
        
        # 确保标签列存在 (如果用的是 Kaggle 数据集，有时需要重命名)
        if config.LABEL_COLUMN not in df.columns:
            # 尝试自动查找常见的标签列名
            candidates = ['Label', 'label', 'Attack', 'attack', 'y']
            for c in candidates:
                if c in df.columns:
                    print(f"Renaming column '{c}' to '{config.LABEL_COLUMN}'")
                    df.rename(columns={c: config.LABEL_COLUMN}, inplace=True)
                    break
        
        if config.LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{config.LABEL_COLUMN}' not found in dataset.")

        # 数据清洗
        df = self.clean_data(df)
        
        # 确保列存在
        static_cols = [c for c in config.STATIC_FEATURES if c in df.columns]
        temporal_cols = [c for c in config.TEMPORAL_FEATURES if c in df.columns]
        
        print(f"Features mapped: {len(static_cols)} Static, {len(temporal_cols)} Temporal")

        # 提取特征并转换为 float32
        X_static = df[static_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        X_temporal = df[temporal_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        
        # 标签处理
        y_raw = df[config.LABEL_COLUMN]
        if y_raw.dtype == 'object':
            y = self.label_encoder.fit_transform(y_raw.astype(str))
            print(f"Classes encoded: {self.label_encoder.classes_}")
        else:
            y = y_raw.values.astype(np.int64)
            
        # 划分训练集和测试集
        X_static_train, X_static_test, X_temporal_train, X_temporal_test, y_train, y_test = train_test_split(
            X_static, X_temporal, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 归一化 (fit on train, transform on test)
        X_static_train = self.scaler_static.fit_transform(X_static_train)
        X_static_test = self.scaler_static.transform(X_static_test)
        
        X_temporal_train = self.scaler_temporal.fit_transform(X_temporal_train)
        X_temporal_test = self.scaler_temporal.transform(X_temporal_test)
        
        print(f"Data prepared. Static shape: {X_static_train.shape}, Temporal shape: {X_temporal_train.shape}")
        
        return (X_static_train, X_temporal_train, y_train), (X_static_test, X_temporal_test, y_test)

def get_dataloaders(train_data, test_data, batch_size):
    train_dataset = IDSDataset(*train_data)
    test_dataset = IDSDataset(*test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader