# Homework 5 实验说明 - 高校分群、ECNU 画像与学科排名预测（任务 8/9/10）

核心实验报告 Notebook：`homework_5/analysis.ipynb`
本作业在 Homework 4 已完成的数据导入基础上，完成以下三项任务：

- 任务 8：结合 ESI 学科排名数据，对全球高校进行分群；并检索与华东师范大学（ECNU）相似的高校。
- 任务 9：通过探索性分析，为华东师范大学构建“学科画像”。
- 任务 10：利用数据建模方式，对各学科构建排名预测模型，并评估效果。

## 数据来源
- 直接读取 Homework 4 已导入的 MySQL 表 `esi_data`。
- 需提供 `config.json`（包含 MySQL 连接信息），Notebook 将通过相对路径 `/config.json` 读取。

config.json 参考：
```json
{
  "mysql_user": "user",
  "mysql_password": "password",
  "mysql_host": "localhost",
  "mysql_db": "esi_db"
}
```

## 依赖包
pandas、numpy、matplotlib、seaborn、scikit-learn、sqlalchemy、pymysql、joblib、pyarrow

## 目录结构（与本任务相关）
```
homework_5/
├── analysis.ipynb          # 本作业主 Notebook
├── README.md               # 本说明文档
├── test.py                 # 测试与 MySQL 数据库的连接
└── results/                # 运行输出（自动生成）
    ├── cache/                       # 数据缓存（feather/parquet）
    ├── institution_clusters.csv     # 任务8：高校分群结果
    ├── cluster_rank_percentile_mean.csv
    ├── similar_to_ECNU.csv          # 任务8：相似高校
    ├── ecnu_subject_profile_compare.csv  # 任务9：ECNU 与全局/同簇均值对比
    └── subject_rank_model_scores_rf.csv   # 任务10：模型评估汇总
```