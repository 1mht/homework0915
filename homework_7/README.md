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

### 前置条件
- 已安装必要的Python包：`pytorch-tabnet`, `torch`, `scikit-learn`, `pandas`, `matplotlib`等
- 数据库配置文件 `config.json` 已放置在 `homework_7/` 目录下

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

## TabNet模型特点

### 优势
1. **可解释性**: TabNet提供特征重要性分析
2. **处理混合数据**: 自动处理数值和类别特征
3. **稀疏注意力**: 使用稀疏注意力机制选择重要特征
4. **端到端训练**: 无需手动特征工程

### 模型参数
- `n_d=64`: 决策层维度
- `n_a=64`: 注意力层维度  
- `n_steps=5`: 决策步骤数
- `gamma=1.5`: 正则化参数
- `cat_emb_dim=8`: 类别特征嵌入维度

## 评估指标

### 排名预测评估
- **MSE (均方误差)**: 惩罚大误差
- **MAE (平均绝对误差)**: 直观反映平均偏差
- **MAPE (平均绝对百分比误差)**: 相对误差度量
- **R2 (决定系数)**: 解释方差比例

### 聚类评估
- **轮廓系数**: 衡量聚类质量
- **余弦相似度**: 寻找相似高校
- **PCA可视化**: 降维展示聚类结果

## 结果分析

### TabNet vs MLP比较
TabNet模型相比传统MLP模型具有以下优势：
1. 更好的特征选择能力
2. 更高的可解释性
3. 对表格数据的专门优化
4. 自动处理类别特征

### 聚类发现
通过聚类分析可以发现：
1. ECNU所在的学术群体特征
2. 相似高校的学科分布模式
3. 不同国家/地区高校的聚类特征

## 注意事项

1. **数据缓存**: 首次运行会创建数据缓存，后续运行会更快
2. **内存使用**: TabNet训练需要较多内存，建议分批训练
3. **可视化**: 确保系统安装了中文字体以正确显示中文标签
4. **比较分析**: 如果存在homework_6的结果，会自动进行模型比较

## 技术栈

- **深度学习**: PyTorch, TabNet
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **数据库**: MySQL, SQLAlchemy

---

*完成时间: 2025年*
*作者: 基于homework_6的改进版本*