import json
import os

# 读取配置信息
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

mysql_user = config['mysql_user']
mysql_password = config['mysql_password']
mysql_host = config['mysql_host']
mysql_db = config['mysql_db']
import pandas as pd
import glob
from sqlalchemy import create_engine

# 1. 读取所有csv，合并为一个DataFrame
all_data = []
for path in glob.glob('download/*.csv'):
    filter_value = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, skiprows=1, skipfooter=1, engine='python')
    df.columns = ['subject_rank', 'institution', 'country_region', 'web_of_science_documents', 'cites', 'cites_per_paper', 'top_papers']
    df['filter_value'] = filter_value
    all_data.append(df)
if not all_data:
    print("未找到 download 文件夹下的 csv 文件，程序终止。")
    exit(1)
df_all = pd.concat(all_data, ignore_index=True)

# 2. 数据类型转换（可选，保证字段类型正确）
df_all['subject_rank'] = df_all['subject_rank'].astype(int)
df_all['web_of_science_documents'] = df_all['web_of_science_documents'].astype(int)
df_all['cites'] = df_all['cites'].astype(int)
df_all['cites_per_paper'] = df_all['cites_per_paper'].astype(float)
df_all['top_papers'] = df_all['top_papers'].astype(int)

# 3. 导入MySQL

# 导入数据
conn_str = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}?charset=utf8mb4"
engine = create_engine(conn_str)
df_all.to_sql('esi_data', engine, if_exists='replace', index=False)
print("数据已成功导入数据库表 esi_data 中。")