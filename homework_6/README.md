# Homework 6：深度学习排名预测与聚类分析

## 任务概述
在homework_5的基础上，使用深度学习方法完成以下任务：

### 任务11：深度学习排名预测模型
- 使用深度学习模型预测各学科排名
- 评估指标：MSE、MAPE、MAE、R²
- 与传统机器学习模型对比

### 任务12：ESI数据聚类分析
- 对ESI数据进行聚类分析
- 发现与华东师范大学（ECNU）类似的学校
- 分析相似性原因

## 项目结构
```
homework_6/
├── deep_learning_ranking.ipynb    # 深度学习排名预测模型
├── clustering_analysis.ipynb      # 聚类分析任务
├── utils.py                       # 工具函数
├── config.json                    # 数据库配置
└── results/                       # 输出结果
    ├── deep_learning_scores.csv   # 深度学习模型评估
    ├── results/deep_learning_results.png # 可视化图表
    ├── similar_schools_dl.csv     # 深度学习聚类结果
    ├── results/institution_clusters_dl.csv # 聚类结果
    └── results/clustering_analysis.png # 可视化图表
```

## 运行步骤

### 1：运行深度学习排名预测
```bash
cd homework_6
python deep_learning_ranking.py
```
**输出结果：**
- `results/deep_learning_scores.csv` - 模型评估指标
- `results/deep_learning_results.png` - 可视化图表

### 2：运行聚类分析
```bash
cd homework_6
python clustering_analysis.py
```
**输出结果：**
- `results/similar_schools_dl.csv` - 相似学校列表
- `results/institution_clusters_dl.csv` - 聚类结果
- `results/clustering_analysis.png` - 可视化图表

### 深度学习模型架构
```python
RankingPredictor 类包含：
- 数值特征处理层（Linear + ReLU + BatchNorm + Dropout）
- 类别特征嵌入层（Embedding）
- 多个隐藏层
- 输出层（回归预测）
```
