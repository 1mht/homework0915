import pymysql
import json
import os  

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

mysql_user = config['mysql_user']
mysql_password = config['mysql_password']
mysql_host = config['mysql_host']
mysql_db = config['mysql_db']

try:
    conn = pymysql.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        charset='utf8mb4'
    )
    print("数据库连接成功！")
    conn.close()
except Exception as e:
    print("数据库连接失败：", e)