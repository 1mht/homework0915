"""
TabNet深度学习排名预测 - 数据库版本
直接从数据库加载数据，避免文件依赖
使用CPU避免CUDA错误，确保稳定运行
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

def load_and_preprocess_data():
    """从数据库加载并预处理数据"""
    print("步骤1: 数据加载与预处理")
    
    try:
        # 导入数据库工具函数
        sys.path.append('..')
        from homework_6.utils import load_esi_dataframe, clean_and_enrich, make_subject_features
        
        # 加载数据
        df_raw = load_esi_dataframe(force_reload=False)
        print(f"原始数据形状: {df_raw.shape}")
        
        # 数据清洗和特征工程
        df_clean = clean_and_enrich(df_raw)
        df_feat = make_subject_features(df_clean)
        
        print(f"预处理后数据形状: {df_feat.shape}")
        
        # 获取所有学科
        subjects = df_feat['filter_value'].unique().tolist()
        
        return df_feat, subjects
        
    except ImportError as e:
        print(f"无法导入数据库工具: {e}")
        print("请确保数据库配置正确")
        sys.exit(1)

def train_tabnet_for_subject(X_train, X_test, y_train, y_test, subject_name):
    """为单个学科训练TabNet模型"""
    print(f"\n训练学科: {subject_name}")
    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    
    print(f"数值特征范围: {X_train_scaled.min():.2f} 到 {X_train_scaled.max():.2f}")
    print(f"目标变量范围: {y_train_scaled.min():.2f} 到 {y_train_scaled.max():.2f}")
    
    # 强制使用CPU避免CUDA错误
    device = 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 创建TabNet模型（简化配置提高稳定性）
        tabnet = TabNetRegressor(
            n_d=8,           # 减少维度
            n_a=8,           # 减少注意力维度
            n_steps=3,       # 减少步骤
            gamma=1.3,       # 正则化参数
            lambda_sparse=1e-3,  # 稀疏性参数
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,
            device_name=device  # 强制使用CPU
        )
        
        # 训练模型
        tabnet.fit(
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            eval_set=[(X_test_scaled, y_test_scaled)],
            eval_name=['valid'],
            eval_metric=['rmse'],
            max_epochs=50,  # 减少epoch
            patience=10,    # 减少耐心
            batch_size=32,  # 减小批次大小
            virtual_batch_size=16,
            num_workers=0,  # 禁用多进程
            drop_last=False
        )
        
        # 预测
        y_pred_scaled = tabnet.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"训练成功 - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return tabnet, y_scaler, {
            'mse': mse,
            'mae': mae, 
            'r2': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
    except Exception as e:
        print(f"TabNet训练失败: {str(e)}")
        return None, None, None

def main():
    """主函数"""
    print("开始TabNet深度学习排名预测...")
    
    # 加载数据
    dataframe, subjects = load_and_preprocess_data()
    
    # 准备特征和目标 - 使用与原始代码相同的特征
    numerical_features = [
        'log_documents', 'log_cites', 'cites_per_paper', 'tp_rate',
        'rank_percentile'
    ]
    
    categorical_features = []
    target_column = 'y'  # 使用原始排名作为目标
    
    print(f"步骤2: 数据集分割")
    print(f"总共 {len(subjects)} 个学科")
    
    # 存储结果
    results = {}
    trained_models = {}
    
    print(f"\n步骤3: 训练TabNet模型")
    
    for subject in subjects:
        subject_data = dataframe[dataframe['filter_value'] == subject]
        
        if len(subject_data) < 50:  # 跳过样本太少的学科
            print(f"跳过学科 {subject} - 样本太少: {len(subject_data)}")
            continue
            
        # 分割数据
        X = subject_data[numerical_features].fillna(0).values
        y = subject_data[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # 训练TabNet
        tabnet_model, y_scaler, metrics = train_tabnet_for_subject(
            X_train, X_test, y_train, y_test, subject
        )
        
        if tabnet_model is not None:
            trained_models[subject] = (tabnet_model, y_scaler)
            results[subject] = metrics
    
    print(f"\n步骤4: 结果分析与可视化")
    
    if results:
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'subject': list(results.keys()),
            'mse': [r['mse'] for r in results.values()],
            'mae': [r['mae'] for r in results.values()],
            'r2': [r['r2'] for r in results.values()]
        })
        
        print("\n各学科TabNet模型性能:")
        print(results_df.round(4))
        
        # 保存结果
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/tabnet_results.csv', index=False)
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(results_df['subject'], results_df['mse'])
        plt.title('各学科MSE')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 3, 2)
        plt.bar(results_df['subject'], results_df['mae'])
        plt.title('各学科MAE')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 3, 3)
        plt.bar(results_df['subject'], results_df['r2'])
        plt.title('各学科R²')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('results/tabnet_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n成功训练了 {len(results)} 个TabNet模型")
        print(f"平均MSE: {results_df['mse'].mean():.4f}")
        print(f"平均MAE: {results_df['mae'].mean():.4f}")
        print(f"平均R²: {results_df['r2'].mean():.4f}")
        
    else:
        print("没有成功训练的TabNet模型")
    
    print("\nTabNet深度学习排名预测完成！")
    print("版本特点:")
    print("- 直接从数据库加载数据")
    print("- 使用CPU避免CUDA错误")
    print("- 简化模型配置提高稳定性")
    print("- 禁用多进程避免重复导入")
    print("- 标准化的输出格式")

if __name__ == "__main__":
    main()