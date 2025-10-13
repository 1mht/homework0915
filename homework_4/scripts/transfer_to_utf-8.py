import os
import chardet

def convert_to_utf8(src_folder):
    for fname in os.listdir(src_folder):
        if fname.endswith('.csv'):
            fpath = os.path.join(src_folder, fname)
            with open(fpath, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
                encoding = result['encoding']
            try:
                content = raw.decode(encoding)
            except Exception as e:
                print(f"{fname} 编码检测为{encoding}但解码失败: {e}")
                continue
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"{fname} 已转换为UTF-8编码")

convert_to_utf8(r'download')