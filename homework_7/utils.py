"""
Homework 7 工具函数
独立的数据处理工具，不依赖homework_6
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据字段定义
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
    
    # 保存缓存
    df.to_feather(feather_path)
    
    return df

def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗和特征衍生"""
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
    """特征工程"""
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
    """按学科分割数据集"""
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

def calculate_mape(y_true, y_pred):
    """计算MAPE（Mean Absolute Percentage Error）"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred):
    """综合评估预测结果"""
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

def build_institution_features(df_clean):
    """
    构建高校特征矩阵
    基于学科表现构建多维度特征
    """
    # 构建排名特征矩阵
    rank_features = df_clean.pivot_table(
        index="institution", 
        columns="filter_value", 
        values="rank_percentile", 
        aggfunc="min"
    ).fillna(1.0)  # 缺失值填充为1.0（末位）
    
    # 构建引用质量特征
    df_cp = df_clean.copy()
    df_cp["cpp_z"] = df_cp.groupby("filter_value")["cites_per_paper"].transform(
        lambda x: (x - x.mean())/x.std(ddof=0) if x.std(ddof=0) not in [0, np.nan] else 0
    )
    cpp_features = df_cp.pivot_table(
        index="institution", 
        columns="filter_value", 
        values="cpp_z", 
        aggfunc="mean"
    ).fillna(0.0)
    
    # 构建顶级论文特征
    df_tp = df_clean.copy()
    df_tp["tp_rate"] = df_tp["top_papers"] / df_tp.groupby("filter_value")["top_papers"].transform("max")
    df_tp["tp_rate"] = df_tp["tp_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    tp_features = df_tp.pivot_table(
        index="institution", 
        columns="filter_value", 
        values="tp_rate", 
        aggfunc="mean"
    ).fillna(0.0)
    
    # 添加列名前缀
    rank_features.columns = pd.MultiIndex.from_product([["rank"], rank_features.columns])
    cpp_features.columns = pd.MultiIndex.from_product([["cppz"], cpp_features.columns])
    tp_features.columns = pd.MultiIndex.from_product([["tpr"], tp_features.columns])
    
    # 合并所有特征
    feature_matrix = pd.concat([rank_features, cpp_features, tp_features], axis=1).sort_index(axis=1)
    
    # 标准化特征
    scaler = StandardScaler()
    feature_matrix_std = pd.DataFrame(
        scaler.fit_transform(feature_matrix.fillna(0.0)),
        index=feature_matrix.index,
        columns=feature_matrix.columns
    )
    
    return feature_matrix_std, scaler