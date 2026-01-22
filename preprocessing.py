import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import config

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
        self.filepath = filepath
        self.scaler_static = MinMaxScaler()
        self.scaler_temporal = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def clean_data(self, df):
        """数据清洗：处理缺失值，转换非数值类型"""
        df = df.copy()
        df.fillna(0, inplace=True)
        
        # 简单处理 IP 地址：转换为整数或直接删除（这里示例为删除，实际可用 ipaddress 库转换）
        cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'] 
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=existing_cols, inplace=True)
        
        return df

    def prepare_data(self):
        print(f"Loading data from {self.filepath}...")
        # 实际读取时可能需要指定 encoding 或 low_memory=False
        df = pd.read_csv(self.filepath) 
        
        # 数据清洗
        df = self.clean_data(df)
        
        # 确保列存在 (处理 config 中定义的列可能在清洗中被删的情况)
        static_cols = [c for c in config.STATIC_FEATURES if c in df.columns]
        temporal_cols = [c for c in config.TEMPORAL_FEATURES if c in df.columns]
        
        # 提取特征
        X_static = df[static_cols].values
        X_temporal = df[temporal_cols].values
        y = df[config.LABEL_COLUMN].values
        
        # 标签编码
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            
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