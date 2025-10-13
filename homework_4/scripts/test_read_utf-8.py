import os
import pandas as pd

src_folder = r'download' 

for fname in os.listdir(src_folder):
    if fname.endswith('.csv'):
        fpath = os.path.join(src_folder, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8')
            print(f"{fname} 读取成功，首行：{df.columns.tolist()}")
        except Exception as e:
            print(f"{fname} 读取失败：{e}")