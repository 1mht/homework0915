#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易学生信息管理系统 - Lab1 实验核心代码
"""

import json
import os

class StudentManager:
    def __init__(self, filename="students.json"):
        self.filename = filename
        self.students = self.load_data()
    
    def load_data(self):
        """加载学生数据"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_data(self):
        """保存数据到文件"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.students, f, ensure_ascii=False, indent=2)
    
    def add_student(self, student_id, name, gender, class_name, phone):
        """添加学生信息"""
        # 检查学号是否已存在
        for student in self.students:
            if student['student_id'] == student_id:
                return False
        
        student = {
            'student_id': student_id,
            'name': name,
            'gender': gender,
            'class_name': class_name,
            'phone': phone
        }
        self.students.append(student)
        self.save_data()
        return True
    
    def query_student(self, student_id):
        """按学号查询学生"""
        for student in self.students:
            if student['student_id'] == student_id:
                return student
        return None
    
    def delete_student(self, student_id):
        """删除学生信息"""
        for i, student in enumerate(self.students):
            if student['student_id'] == student_id:
                del self.students[i]
                self.save_data()
                return True
        return False
    
    def display_all(self):
        """显示所有学生信息"""
        if not self.students:
            print("暂无学生信息")
            return
        
        print(f"{'学号':<10} {'姓名':<8} {'性别':<4} {'班级':<12} {'电话':<15}")
        print("-" * 50)
        for student in self.students:
            print(f"{student['student_id']:<10} {student['name']:<8} "
                  f"{student['gender']:<4} {student['class_name']:<12} "
                  f"{student['phone']:<15}")

def main():
    """主函数"""
    manager = StudentManager()
    
    while True:
        print("\n=== 学生信息管理系统 ===")
        print("1. 添加学生")
        print("2. 查询学生")
        print("3. 删除学生")
        print("4. 显示所有")
        print("5. 退出")
        
        choice = input("请选择操作: ").strip()
        
        if choice == '1':
            student_id = input("学号: ")
            name = input("姓名: ")
            gender = input("性别(男/女): ")
            class_name = input("班级: ")
            phone = input("电话: ")
            
            if manager.add_student(student_id, name, gender, class_name, phone):
                print("添加成功!")
            else:
                print("添加失败，学号已存在!")
                
        elif choice == '2':
            student_id = input("请输入学号: ")
            student = manager.query_student(student_id)
            if student:
                print(f"找到学生: {student['name']}")
                print(f"详细信息: {student}")
            else:
                print("未找到该学生")
                
        elif choice == '3':
            student_id = input("请输入要删除的学号: ")
            if manager.delete_student(student_id):
                print("删除成功!")
            else:
                print("删除失败，学号不存在!")
                
        elif choice == '4':
            manager.display_all()
            
        elif choice == '5':
            print("谢谢使用!")
            break
            
        else:
            print("无效选择!")

if __name__ == "__main__":
    main()