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
        print(f"读取文件时出错：{e}")
        return []

def save_students_data(data, filename="students.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            print("success")
            return True
    except Exception as e:
        print(f"error:{e}")
        return False
    
# PART.2 read(singly/all), write, delete
def  read_singly(student_id, filename="students.json"):
    students = load_students_data(filename)
    for student in students:
        if student["id"] == student_id:
            return student
    return None

def read_all(filename="students.json"):
    students = load_students_data(filename)
    if not students:
        print("无学生信息")
        return []
    print(f"\n============== 所有学生信息（共 {len(students)} 人）=============")
    print(f"{'学号':<12} {'姓名':<6} {'性别':<4} {'班级':<8} {'电话':<15}")
    print("-" * 50)
    for student in students:
        gender = "男" if student["sex"] == 1 else "女"
        print(f"{student['id']:<15} {student['name']:<8} "
              f"{gender:<4} {student['room_id']:<8} "
              f"{student['tele']:<15}")
    return students

def id_exists(students, student_id):
    for student in students:
        if student['id'] == student_id:
            return True
    return False

def tele_exists(students, tele):
    for student in students:
        if student['tele'] == tele:
            return True
    return False

def write():
    print("\n=== 增加新学生 ===")
    students = load_students_data()

    while True:
        student_id_input = input("请输入学号：").strip()
        if not student_id_input.isdigit():
            print("请输入正确学号！")
        elif id_exists(students, student_id_input):
            print("学号已存在！")
        else:
            student_id = student_id_input
            break

    name = input("请输入姓名：").strip()

    while True:
        sex_input = input("请输入性别：").strip()
        if sex_input == "男":
            sex = int(1)
            break
        elif sex_input == "女":
            sex = int(0)
            break
        else:
            print("请输入男或女！不要输入其他字符")

    while True:
        room_id_input = input("请输入宿舍号：").strip()
        if not room_id_input.isdigit():
            print("请输入正确宿舍号！")
        else:
            room_id = room_id_input
            break

    while True:
        tele_input = input("请输入电话号码：").strip()
        if not tele_input.isdigit():
            print("请输入正确电话号码！")
        elif tele_exists(students, tele_input):
            print("电话号码已存在！")
        else:
            tele = tele_input
            break
    
    new_student = {
        "id": student_id,
        "name": name,
        "sex": sex,
        "room_id": room_id,
        "tele": tele
    }
    
    students.append(new_student)
    if save_students_data(students):
        print("学生增加成功！")
        return True
    else:
        print("学生保存失败！")
        return False

# PART.3 main function with 3 chioces
def main():
    while True:
        print("\n" + "="*40)
        print("            学生信息管理系统")
        print("="*40)
        print("1. 查询学生 (select)")
        print("2. 添加学生 (write)") 
        print("3. 显示所有学生 (read)")
        print("4. 退出系统")
        print("="*40)

        choice = input("请键入相应数字：").strip()
        if choice == "1":
            print("\n== 查询学生 ===")
            while True:
                student_id_input = input("请输入要查询的学生的学号：").strip()
                if student_id_input.isdigit():
                    student = read_singly(student_id_input)
                    if student:
                        print(f"\n找到学生信息:")
                        print(f"学号: {student['id']}")
                        print(f"姓名: {student['name']}")
                        print(f"性别: {'男' if student['sex'] == 1 else '女'}")
                        print(f"班级: {student['room_id']}")
                        print(f"电话: {student['tele']}")
                    else:
                        print(f"未找到学号为 {student_id_input} 的学生")
                    break
                else:
                    print("请输入正确的学号！")
        
        elif choice == '2':
            write()
        elif choice == '3':
            read_all()
        elif choice == '4':
            print("thank you!")
            break

        else:
            print("请输入正确字符！")
        
        input("\n按回车键继续……")

if __name__ == "__main__":
    main()