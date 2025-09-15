import json
import os

GENDER_MALE = 1
GENDER_FEMALE = 0

# PART.1 load and save data
def load_students_data(filename="students.json"):
    if not os.path.exists(filename):
        print(f"file {filename} do not exist, create empty data")
        return []
    try:
        with open(filename, 'r', encoding = 'utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            else:
                print("warning: 格式错误, return empty list")
                return []
    except json.JSONDecodeError:
        print("JSON格式错误，返回空列表")
        return []
    except Exception as e:
        print("读取文件时出错：{e}")
        return []

def save_students_data(filename="stdents.json", data):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            print("success")
            return True
    except Exception as e:
        print(f"error:{e}")
        return False
    

def main():
    print("=== students management system ===")
    print("=== 系统准备就绪 ===")

if __name__ == "__main__":
    main()