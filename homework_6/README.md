# Homework 6：深度学习排名预测与聚类分析


## 我怎么理解这两个任务

### 任务 11：深度学习排名预测模型（按学科）
- 用深度学习把“学科相关的输入特征”映射到“学科排名（或百分位）”。
- 选了 PyTorch 写一个 MLP（多层感知机），因为它上手快、改网络结构方便。
- 评估指标主要是 MSE、MAPE、MAE、R²：一起看更全面，避免只盯着一个数。

### 任务 12：ESI 数据聚类 + 找相似高校
- 先把每个高校在不同学科上的表现（排名百分位、被引强度、顶尖论文率等）做成一个特征向量。
- 用 KMeans 在这个向量空间聚类；再用余弦相似度找出和 ECNU 最像的高校。
- 画图帮助理解每个簇的分布和 ECNU 附近的学校长什么样。

---

## 运行步骤

前置：`config.json` 放在 `homework_6/` 目录下（已提供示例）

1) 运行任务 11：深度学习排名预测
```powershell
python .\deep_learning_ranking.py
```
输出会保存在 `results/` 目录下：
- `results/deep_learning_scores.csv`：按学科汇总的 MSE / RMSE / MAE / MAPE / R²
- `results/deep_learning_results.png`：指标分布、训练曲线、预测 vs 真实散点

1) 运行任务 12：聚类与相似高校
```powershell
python .\clustering_analysis.py
```
输出会保存在 `results/` 目录下：
- `results/institution_clusters_dl.csv`：每所高校的聚类标签
- `results/similar_schools_dl.csv`：与 ECNU 最相似的高校列表（带相似度与簇）
- `results/clustering_analysis.png`：PCA 降维后的聚类可视化 + 若干分布图

---

## 项目结构
```
homework_6/
├── report.md                       # 主要的实验报告md文件
├── deep_learning_ranking.py        # 任务11：深度学习排名预测（PyTorch）
├── clustering_analysis.py          # 任务12：ESI 聚类与相似高校
├── utils.py                        # 数据读取与特征工程工具
├── config.json                     # 数据库配置（mysql）
└── results/                        # 输出目录
        ├── deep_learning_scores.csv
        ├── deep_learning_results.png
        ├── similar_schools_dl.csv
        ├── institution_clusters_dl.csv
        └── clustering_analysis.png
```

> 说明：`deep_learning_ranking.py` 默认只训练前 5 个学科用于演示（避免跑太久）。你可以在代码里把 `subjects_to_train` 改成更多学科。

---

## 我在做题时遇到的关键点

1) 数据怎么来？
- utils.py 会用 `config.json` 里的 MySQL 连接（`esi_data` 表），并做清洗/特征工程。

1) 任务 11 模型怎么搭？
- 数值特征：log_documents、log_cites、cites_per_paper、tp_rate。
- 类别特征：country_region（用简单的类别编码+Embedding）。
- 我用 MSELoss 训练，最后用 MSE/MAE/MAPE/R² 综合评估。

1) 任务 12 聚类怎么做？
- 先把“高校×学科”的特征矩阵拼好，再整体标准化。
- KMeans 的 k 我用轮廓系数和肘部法做参考，代码里会自动给一个推荐值。
- 相似度用余弦相似度，结果里会看到与 ECNU 最接近的高校及其簇。

1) 我的一点反思
- 排名/百分位最好在学科内比较，跨学科差异很大。
- 评估不要只看一个指标，MAPE 对小值敏感，R² 能帮助判断拟合程度。
- 聚类没有“标准答案”，要结合业务语境解释每个簇的含义。
- 中文图例乱码：本作业选择了新设置的conda环境，matplotlib画图又出现中文乱码的情况，按照过往解决思路：`修改字体文件->删缓存->在 clustering_analysis.py 增加以下代码显性显示中文字体：`
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

