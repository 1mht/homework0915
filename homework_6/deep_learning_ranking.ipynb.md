<jupyter_text>
Homework 6：深度学习排名预测模型本Notebook使用深度学习方法构建学科排名预测模型，并与传统机器学习模型进行对比。
<jupyter_code>
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义工具函数
from utils import (
    load_esi_dataframe, clean_and_enrich, make_subject_features,
    per_subject_split, ESIDataset, calculate_mape, evaluate_predictions
)

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
<jupyter_output>
<empty_output>
<jupyter_text>
1. 数据加载与预处理复用homework_5的数据处理流程，确保数据一致性。
<jupyter_code>
# 加载数据
df_raw = load_esi_dataframe(force_reload=False)
print(f"原始数据形状: {df_raw.shape}")

# 数据清洗和特征衍生
df_clean = clean_and_enrich(df_raw)
print(f"清洗后数据形状: {df_clean.shape}")

# 特征工程
df_feat = make_subject_features(df_clean)
print(f"特征工程后数据形状: {df_feat.shape}")

# 显示数据基本信息
print("\n数据基本信息:")
print(df_feat[['log_documents', 'log_cites', 'cites_per_paper', 'tp_rate', 'y']].describe())
<jupyter_output>
<empty_output>
<jupyter_text>
2. 深度学习模型设计设计一个多层感知机（MLP）模型用于排名预测。
<jupyter_code>
class RankingPredictor(nn.Module):
    """
    深度学习排名预测模型
    多层感知机（MLP）架构
    """
    
    def __init__(self, num_numerical_features, num_categorical_features, hidden_sizes=[64, 32, 16]):
        super(RankingPredictor, self).__init__()
        
        # 数值特征处理层
        self.numerical_layers = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.2)
        )
        
        # 类别特征嵌入层（简化处理）
        self.categorical_embedding = nn.Embedding(
            num_embeddings=num_categorical_features * 100,  # 假设最多100个类别
            embedding_dim=8
        )
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        input_size = hidden_sizes[0] + 8  # 数值特征 + 类别嵌入
        
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(input_size, hidden_sizes[i+1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_sizes[i+1]),
                    nn.Dropout(0.2)
                )
            )
            input_size = hidden_sizes[i+1]
        
        # 输出层
        self.output_layer = nn.Linear(input_size, 1)
        
    def forward(self, numerical_features, categorical_features):
        # 处理数值特征
        x_numerical = self.numerical_layers(numerical_features)
        
        # 处理类别特征（简单平均池化）
        cat_embedded = self.categorical_embedding(categorical_features)
        x_categorical = cat_embedded.mean(dim=1)
        
        # 合并特征
        x = torch.cat([x_numerical, x_categorical], dim=1)
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 输出
        output = self.output_layer(x)
        return output
<jupyter_output>
<empty_output>
<jupyter_text>
3. 模型训练函数定义训练和评估函数。
<jupyter_code>
def train_model_for_subject_dl(train_data, val_data, num_epochs=100, learning_rate=0.001):
    """
    为单个学科训练深度学习模型
    """
    # 定义特征
    numerical_features = ["log_documents", "log_cites", "cites_per_paper", "tp_rate"]
    categorical_features = ["country_region"]
    
    # 创建数据集
    train_dataset = ESIDataset(train_data, numerical_features, categorical_features)
    val_dataset = ESIDataset(val_data, numerical_features, categorical_features)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = RankingPredictor(
        num_numerical_features=len(numerical_features),
        num_categorical_features=len(categorical_features)
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_num, (numerical, categorical, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(numerical, categorical)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for numerical, categorical, targets in val_loader:
                outputs = model(numerical, categorical)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses

def evaluate_model_dl(model, test_data):
    """
    评估深度学习模型
    """
    numerical_features = ["log_documents", "log_cites", "cites_per_paper", "tp_rate"]
    categorical_features = ["country_region"]
    
    test_dataset = ESIDataset(test_data, numerical_features, categorical_features)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for numerical, categorical, targets in test_loader:
            outputs = model(numerical, categorical)
            predictions.extend(outputs.numpy().flatten())
            true_values.extend(targets.numpy().flatten())
    
    return np.array(predictions), np.array(true_values)
<jupyter_output>
<empty_output>
<jupyter_text>
4. 按学科训练深度学习模型对每个学科分别训练深度学习模型。
<jupyter_code>
# 数据集分割
splits = per_subject_split(df_feat, random_state=42)
print(f"总共 {len(splits)} 个学科")

# 训练深度学习模型
dl_results = {}
dl_models = {}

for subject, data_parts in list(splits.items())[:10]:  # 先测试前10个学科
    if len(data_parts["train"]) < 20 or len(data_parts["test"]) < 10:
        continue
    
    print(f"\n训练学科: {subject}")
    print(f"训练样本: {len(data_parts['train'])}, 测试样本: {len(data_parts['test'])}")
    
    try:
        # 训练模型
        model, train_losses, val_losses = train_model_for_subject_dl(
            data_parts["train"], data_parts["val"], num_epochs=50
        )
        
        # 评估模型
        predictions, true_values = evaluate_model_dl(model, data_parts["test"])
        
        # 计算评估指标
        metrics = evaluate_predictions(true_values, predictions)
        
        # 保存结果
        dl_results[subject] = {
            "metrics": metrics,
            "predictions": predictions,
            "true_values": true_values,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        dl_models[subject] = model
        
        print(f"评估结果: {metrics}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        continue

print(f"\n成功训练了 {len(dl_results)} 个学科的深度学习模型")
<jupyter_output>
<empty_output>
<jupyter_text>
5. 模型性能评估与可视化
<jupyter_code>
# 汇总所有学科的评估结果
if dl_results:
    metrics_df = pd.DataFrame([{
        "subject": subject,
        "MSE": result["metrics"]["MSE"],
        "RMSE": result["metrics"]["RMSE"], 
        "MAE": result["metrics"]["MAE"],
        "MAPE": result["metrics"]["MAPE"],
        "R2": result["metrics"]["R2"]
    } for subject, result in dl_results.items()])
    
    print("深度学习模型性能汇总:")
    print(metrics_df.describe())
    
    # 保存结果
    metrics_df.to_csv("results/deep_learning_scores.csv", index=False)
    
    # 可视化评估指标分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    for i, metric in enumerate(metrics_to_plot):
        row, col = i // 3, i % 3
        axes[row, col].hist(metrics_df[metric], bins=20, alpha=0.7, color='skyblue')
        axes[row, col].set_title(f'{metric} Distribution')
        axes[row, col].set_xlabel(metric)
        axes[row, col].set_ylabel('Frequency')
    
    # 训练损失曲线示例
    if len(dl_results) > 0:
        example_subject = list(dl_results.keys())[0]
        train_losses = dl_results[example_subject]["train_losses"]
        val_losses = dl_results[example_subject]["val_losses"]
        
        axes[1, 2].plot(train_losses, label='Train Loss')
        axes[1, 2].plot(val_losses, label='Val Loss')
        axes[1, 2].set_title(f'Training Curve ({example_subject})')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 预测vs真实值散点图示例
    if len(dl_results) > 0:
        example_subject = list(dl_results.keys())[0]
        predictions = dl_results[example_subject]["predictions"]
        true_values = dl_results[example_subject]["true_values"]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predictions, alpha=0.6)
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
        plt.xlabel('True Rank')
        plt.ylabel('Predicted Rank')
        plt.title(f'Predictions vs True Values ({example_subject})')
        plt.show()
else:
    print("没有成功训练的模型")
<jupyter_output>
<empty_output>
<jupyter_text>
6. 与传统机器学习模型对比（可选）与homework_5的随机森林模型进行对比。
<jupyter_code>
# 这里可以添加与传统模型的对比代码
# 需要先运行homework_5的模型来获取对比数据

print("深度学习模型训练完成！")
print("主要评估指标:")
print("- MSE: 均方误差，值越小越好")
print("- RMSE: 均方根误差，值越小越好") 
print("- MAE: 平均绝对误差，值越小越好")
print("- MAPE: 平均绝对百分比误差，值越小越好")
print("- R2: 决定系数，越接近1越好")