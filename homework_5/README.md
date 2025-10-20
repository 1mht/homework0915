# Homework 5 实验说明 - 高校分群、ECNU 画像与学科排名预测（任务 8/9/10）

本作业在 Homework 4 已完成的数据导入基础上，仅聚焦以下三项任务：

- 任务 8：结合 ESI 学科排名数据，对全球高校进行分群；并检索与华东师范大学（ECNU）相似的高校。
- 任务 9：通过探索性分析，为华东师范大学构建“学科画像”。
- 任务 10：利用数据建模方式，对各学科构建排名预测模型，并评估效果。

核心 Notebook：`homework_5/analysis.ipynb`

## 数据来源
- 直接读取 Homework 4 已导入的 MySQL 表 `esi_data`。
- 需提供 `homework_4/config.json`（包含 MySQL 连接信息），Notebook 将通过相对路径 `../homework_4/config.json` 读取。
- 若路径不适用，可设置环境变量 `CFG_PATH_OVERRIDE` 指向该配置文件绝对路径。

config.json 参考：
```json
{
  "mysql_user": "user",
  "mysql_password": "password",
  "mysql_host": "localhost",
  "mysql_db": "esi_db"
}
```

## 环境与依赖
- Python 3.9+（建议）
- 依赖包：pandas、numpy、matplotlib、seaborn、scikit-learn、sqlalchemy、pymysql、joblib、pyarrow
- 可直接使用本目录下的 `requirements.txt` 安装

## 目录结构（与本任务相关）
```
homework_5/
├── analysis.ipynb          # 本作业主 Notebook（仅包含任务 8/9/10）
├── README.md               # 本说明文档
├── requirements.txt        # 依赖清单
└── results/                # 运行输出（自动生成）
    ├── cache/                       # 数据缓存（feather/parquet）
    ├── institution_clusters.csv     # 任务8：高校分群结果
    ├── cluster_rank_percentile_mean.csv
    ├── similar_to_ECNU.csv          # 任务8：相似高校
    ├── ecnu_subject_profile_compare.csv  # 任务9：ECNU 与全局/同簇均值对比
    ├── subject_rank_model_scores_rf.csv   # 任务10：模型评估汇总
    └── models_/rf/*.joblib          # （可选）持久化模型
```

## 运行步骤
1) 打开并依序运行 `analysis.ipynb`：
   - 第 1 节：连接数据库并加载 `esi_data`（首次会缓存到 `results/cache/`）。
   - 第 2-5 节：数据清洗、特征构建、聚类与可视化（任务 8）。
   - 第 6-7 节：近邻检索相似高校、ECNU 画像（任务 8/9）。
   - 第 8-15 节：学科排名预测的特征工程、切分、训练、评估与导出（任务 10）。
2) 输出文件将写入 `homework_5/results/`，可直接用于报告与复查。

## 技术要点摘要
- 高校×学科特征：rank_percentile（小优）、cites_per_paper z 分数、top_papers 学科内占比；整体 StandardScaler。
- 聚类：KMeans(k=3..12) 通过轮廓系数 / Calinski-Harabasz 选 k；t-SNE/PCA 可视化。
- 相似高校：标准化特征空间上用余弦相似度的近邻检索，支持按国家过滤。
- 排名预测：每学科独立建模（RandomForest / HistGB），评估 MAE / RMSE / Spearman，支持超参搜索。

## 常见问题
- 配置找不到：设置 `CFG_PATH_OVERRIDE` 为 `config.json` 的绝对路径后重试。
- 图表中文：可在 Notebook 顶部设置中文字体（SimHei / Microsoft YaHei）。

## 致谢
本任务基于 Homework 4 的数据准备与清洗成果，仅保留并延展 8/9/10 三项分析与建模内容。
