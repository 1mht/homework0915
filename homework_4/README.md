# Lab4 实验报告 - ESI学科数据导入数据库与查询

## 1. 作业说明
实现了优化schema并导入 MySQL 数据库，进行作业中的数据查询。

主要脚本：
- preprocess.py：数据预处理与导入 MySQL 脚本
- SQLscripts/：建表与索引 SQL 脚本
- results/：查询结果导出

## 2. 项目结构

```
homework_4/
├── preprocess.py                  # 数据预处理与导入 MySQL 脚本
├── README.md                      # 项目说明文档
├── test.py                        # 测试与 MySQL 数据库的连接
├── config.json                    # 连接 MySQL 配置文件 *ignore
├── results/                       # 查询结果导出
│    ├── esi_data_ecnu_rankings.csv     # 华东师范大学各学科排名
│    ├── esi_data_mainland_rankings.csv # 中国大陆大学表现
│    └── esi_data_region_rankings.csv   # 全球区域学科表现
├── scripts/                       # csv原始数据转码UTF-8及测试脚本与数据
│    ├── test_read_utf-8.py   
│    ├── ecnu_analysis.py
│    ├── transfer_to_utf-8.py
│    └── ecnu_analysis_results.csv
├── SQLscripts/                    # 建表与索引 SQL 脚本
│    ├── create.sql                     # 建表语句
│    ├── create_with_index.sql          # 建表+索引语句
│    └── add_index.sql                  # 已有表增加索引语句
└── download/                      # 学科领域原始数据
```

## 3. 环境配置

### 1. MySQL 数据库
- 建表脚本见 SQLscripts/create.sql 和 add_index.sql 或 create_with_index.sql。
- 完善 config.json 并放入homework_4文件夹下 (使用是localhost)
```json
{
  "mysql_user": "user",
  "mysql_password": "password",
  "mysql_host": "localhost",
  "mysql_db": "esi_db"
}
```

### 2. Python 库安装
推荐使用 pip 安装依赖库：
```bash
pip install pandas sqlalchemy pymysql
```

## 3. 数据库结构说明

- 表名：esi_data
- 主键：filter_value + subject_rank（联合主键）
- 主要字段：
  - subject_rank BIGINT
  - institution VARCHAR(255)
  - country_region VARCHAR(255)
  - web_of_science_documents BIGINT
  - cites BIGINT
  - cites_per_paper DOUBLE
  - top_papers BIGINT
  - filter_value VARCHAR(255)

## 4. SQL脚本说明
- create.sql：标准建表语句，含联合主键
- add_index.sql：已有表增加索引，提升 institution/country_region/filter_value 查询效率
- create_with_index.sql：建表并加速索引，适合大数据量分析

## 5. 查询与分析示例
- 查询华东师范大学各学科排名
- 查询中国大陆大学各学科表现
- 分析全球不同区域学科表现

## 6.实验收获
- 学习了导入 MySQL数据库的基本操作，一些简单的查询脚本
- 简单理解了数据预处理的逻辑，学会了设置一个合适的schema
