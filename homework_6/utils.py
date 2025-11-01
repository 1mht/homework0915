"""
Homework 6 工具函数
复用 homework_5 的数据处理逻辑，添加深度学习相关功能
"""

import os
import json
import pathlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# 数据字段定义（与homework_5保持一致）
DATA_FIELDS = [
    "subject_rank", "institution", "country_region", 
    "web_of_science_documents", "cites", "cites_per_paper", 
    "top_papers", "filter_value"
]

def load_db_config():
    """从当前工作目录读取 config.json"""
    p = (pathlib.Path.cwd() / "config.json").resolve()
    if p.exists():
        print(f"Using config: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError("未找到配置文件。请将 config.json 放在当前工作目录下。")

def get_engine(cfg):
    """创建数据库连接引擎"""
    user = cfg["mysql_user"]
    pwd = cfg["mysql_password"]
    host = cfg["mysql_host"]
    db = cfg["mysql_db"]
    url = f"mysql+pymysql://{user}:{pwd}@{host}/{db}?charset=utf8mb4"
    return create_engine(url)

def load_esi_dataframe(force_reload=False):
    """加载ESI数据，支持缓存"""
    CACHE_DIR = pathlib.Path("results/cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feather_path = CACHE_DIR / "esi_data.feather"
    
    if not force_reload and feather_path.exists():
        try:
            return pd.read_feather(feather_path)
        except Exception:
            pass
    
    cfg = load_db_config()
    engine = get_engine(cfg)
    sql = "SELECT " + ",".join(DATA_FIELDS) + " FROM esi_data"
    df = pd.read_sql(sql, engine)
    return df

def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗和特征衍生（与homework_5保持一致）"""
    d = df.copy()
    
    # 去重
    d = d.drop_duplicates()
    
    # 文本标准化
    for col in ["institution", "country_region", "filter_value"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()
    
    d["country_region"] = d["country_region"].replace(["", "nan", "None"], "UNKNOWN")
    
    # 数值字段类型转换
    for col in ["subject_rank", "web_of_science_documents", "cites", "top_papers"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    
    if "cites_per_paper" in d.columns:
        d["cites_per_paper"] = pd.to_numeric(d["cites_per_paper"], errors="coerce")
    
    # 删除缺少关键字段的行
    d = d.dropna(subset=["subject_rank", "institution", "filter_value"])
    
    # 计算每学科的 max_rank 和 rank_percentile
    d["max_rank"] = d.groupby("filter_value")["subject_rank"].transform("max")
    d["rank_percentile"] = (d["subject_rank"] - 1) / (d["max_rank"].replace(1, np.nan) - 1)
    d["rank_percentile"] = d["rank_percentile"].fillna(0.0).clip(0, 1)
    
    return d

def make_subject_features(d: pd.DataFrame):
    """特征工程（与homework_5保持一致）"""
    d = d.copy()
    
    # 对数变换处理偏态分布
    d["log_documents"] = np.log1p(d["web_of_science_documents"])
    d["log_cites"] = np.log1p(d["cites"])
    
    # 高被引论文率
    d["tp_rate"] = d["top_papers"] / d.groupby("filter_value")["top_papers"].transform("max")
    d["tp_rate"] = d["tp_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    
    # 目标变量
    d["y"] = d["subject_rank"].astype(float)
    d["y_percentile"] = d["rank_percentile"].astype(float)
    
    return d

def per_subject_split(d: pd.DataFrame, random_state=42):
    """按学科分割数据集（与homework_5保持一致）"""
    from math import floor, ceil
    
    splits = {}
    for subject, sdf in d.groupby("filter_value"):
        sdf = sdf.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(sdf)
        
        if n < 5:
            continue
            
        n_train = floor(0.6 * n)
        n_test = ceil(0.2 * n)
        
        idx_train = sdf.index[:n_train]
        idx_test = sdf.index[-n_test:]
        idx_val = sdf.index[n_train:n - n_test]  # 验证集
        
        splits[subject] = {
            "train": sdf.loc[idx_train],
            "val": sdf.loc[idx_val],
            "test": sdf.loc[idx_test],
        }
    
    return splits

class ESIDataset(Dataset):
    """PyTorch数据集类，用于深度学习模型训练"""
    
    def __init__(self, dataframe, numerical_features, categorical_features, target_col="y"):
        self.data = dataframe
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        
        # 数值特征标准化
        self.scaler = StandardScaler()
        self.numerical_data = self.scaler.fit_transform(
            self.data[self.numerical_features].fillna(0).values
        )
        
        # 类别特征编码（简单数值化）
        self.categorical_data = self.data[self.categorical_features].astype('category')
        self.cat_codes = {}
        for col in self.categorical_features:
            self.cat_codes[col] = self.categorical_data[col].cat.codes.values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 数值特征
        numerical = torch.FloatTensor(self.numerical_data[idx])
        
        # 类别特征（转换为one-hot或直接使用编码）
        categorical = torch.LongTensor([self.cat_codes[col][idx] for col in self.categorical_features])
        
        # 目标变量
        target = torch.FloatTensor([self.data.iloc[idx][self.target_col]])
        
        return numerical, categorical, target

def calculate_mape(y_true, y_pred):
    """计算MAPE（Mean Absolute Percentage Error）"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred):
    """综合评估预测结果"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }