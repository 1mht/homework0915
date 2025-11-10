# Homework 7：TabNet深度学习排名预测与聚类分析

## 项目概述

本作业在homework_6的基础上，使用TabNet深度学习模型重新训练排名预测模型，并完成聚类分析任务。

### 主要改进

1. **TabNet模型**: 使用先进的TabNet深度学习模型替代传统的MLP模型
2. **独立代码**: homework_7完全独立，不依赖homework_6的代码
3. **增强功能**: 包含特征重要性分析和模型比较

## 项目结构

```
homework_7/
├── README.md                       # 项目说明文档
├── report.md                       # 项目实验报告
├── tabnet_ranking.py               # 任务11：TabNet深度学习排名预测
├── clustering_analysis.py          # 任务12：ESI聚类与相似高校分析
├── utils.py                        # 独立的数据处理工具函数
├── config.json                     # 数据库配置文件
└── results/                        # 输出目录
        ├── tabnet_scores.csv       # TabNet模型评估结果
        ├── tabnet_results.png      # TabNet结果可视化
        ├── similar_schools_dl.csv  # 相似高校列表
        ├── institution_clusters_dl.csv  # 聚类结果
        └── clustering_analysis.png # 聚类可视化
```

## 运行步骤

### 1. 运行TabNet深度学习排名预测

```bash
cd homework_7
python tabnet_ranking.py
```

**输出文件:**
- `results/tabnet_scores.csv`: 各学科的MSE/RMSE/MAE/MAPE/R2评估指标
- `results/tabnet_results.png`: 模型性能可视化和特征重要性分析

### 2. 运行聚类分析

```bash
cd homework_7
python clustering_analysis.py
```

**输出文件:**
- `results/similar_schools_dl.csv`: 与ECNU最相似的高校列表
- `results/institution_clusters_dl.csv`: 所有高校的聚类标签
- `results/clustering_analysis.png`: 聚类结果可视化