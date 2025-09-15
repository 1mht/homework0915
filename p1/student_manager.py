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

def save_students_data(data, filename="stdents.json"):
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
    
    # 测试数据加载和保存
    print("\n1. 测试数据加载...")
    data = load_students_data()
    print(f"加载到 {len(data)} 条学生记录")
    
    # 创建测试数据
    test_student = {
        "id": "1",
        "name": "测试学生", 
        "sex": 1,
        "room_id": 101,
        "tele": "13800138000"
    }
    
    print("\n2. 测试数据保存...")
    test_data = [test_student]
    if save_students_data(test_data, "test.json"):
        print("测试数据保存成功！")
        
        # 测试重新加载
        print("\n3. 测试重新加载...")
        reloaded_data = load_students_data("test.json")
        print(f"重新加载到 {len(reloaded_data)} 条记录")
        if reloaded_data:
            print(f"第一条记录: {reloaded_data[0]}")
    else:
        print("数据保存失败！")
    
    # 清理测试文件
    import os
    if os.path.exists("test.json"):
        os.remove("test.json")
        print("测试文件已清理")

if __name__ == "__main__":
    main()