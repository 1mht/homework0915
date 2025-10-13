"""
ECNU数据分析脚本
从downloads文件夹中的CSV文件提取ECNU相关数据
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Optional


class ECNUDataAnalyzer:
    """ECNU数据分析器"""
    
    def __init__(self, downloads_path: str = "downloads"):
        """
        初始化分析器
        
        Args:
            downloads_path: downloads文件夹路径
        """
        self.downloads_path = downloads_path
        self.results = []
    
    def extract_research_field(self, first_line: str) -> Optional[str]:
        """
        从第一行提取研究领域
        
        Args:
            first_line: CSV文件的第一行
            
        Returns:
            研究领域名称，如果未找到则返回None
        """
        # 查找 "Filter Value(s):" 后面的内容
        if "Filter Value(s):" in first_line:
            start_idx = first_line.find("Filter Value(s):") + len("Filter Value(s):")
            # 查找下一个逗号或行尾
            end_idx = first_line.find(",", start_idx)
            if end_idx == -1:
                end_idx = len(first_line)
            
            field = first_line[start_idx:end_idx].strip()
            return field
        return None
    
    def process_csv_file(self, file_path: str) -> Optional[Dict]:
        """
        处理单个CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            包含ECNU数据的字典，如果未找到ECNU则返回None
        """
        try:
            # 读取第一行获取研究领域
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            research_field = self.extract_research_field(first_line)
            if not research_field:
                print(f"警告: 无法从文件 {file_path} 提取研究领域")
                return None
            
            # 读取数据部分（跳过第一行）
            df = pd.read_csv(file_path, skiprows=1)
            
            # 查找ECNU数据
            ecnu_data = df[df['Institutions'] == 'EAST CHINA NORMAL UNIVERSITY']
            
            if ecnu_data.empty:
                print(f"信息: 在 {research_field} 领域未找到ECNU数据")
                return None
            
            # 提取ECNU数据
            ecnu_row = ecnu_data.iloc[0]
            result = {
                'research_field': research_field,
                'institution': ecnu_row['Institutions'],
                'country_region': ecnu_row['Countries/Regions'],
                'web_of_science_documents': ecnu_row['Web of Science Documents'],
                'cites': ecnu_row['Cites'],
                'cites_per_paper': ecnu_row['Cites/Paper'],
                'top_papers': ecnu_row['Top Papers'],
                'file_name': os.path.basename(file_path)
            }
            
            print(f"找到ECNU在 {research_field} 领域的数据")
            return result
            
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")
            return None
    
    def analyze_all_files(self) -> List[Dict]:
        """
        分析所有CSV文件
        
        Returns:
            包含所有ECNU数据的列表
        """
        csv_files = glob.glob(os.path.join(self.downloads_path, "*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        self.results = []
        
        for csv_file in sorted(csv_files):
            result = self.process_csv_file(csv_file)
            if result:
                self.results.append(result)
        
        print(f"总共找到 {len(self.results)} 个包含ECNU数据的文件")
        return self.results
    
    def save_results_to_csv(self, output_file: str = "ecnu_analysis_results.csv"):
        """
        将结果保存到CSV文件
        
        Args:
            output_file: 输出文件名
        """
        if not self.results:
            print("没有数据可保存")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"结果已保存到 {output_file}")
    
    def print_summary(self):
        """打印分析摘要"""
        if not self.results:
            print("没有找到ECNU数据")
            return
        
        print("\n" + "="*80)
        print("ECNU数据分析摘要")
        print("="*80)
        
        df = pd.DataFrame(self.results)
        
        print(f"\n总共在 {len(self.results)} 个研究领域中找到ECNU数据:")
        for field in df['research_field'].unique():
            field_data = df[df['research_field'] == field]
            print(f"  - {field}")
        
        print(f"\n总Web of Science文档数: {df['web_of_science_documents'].sum()}")
        print(f"总引用次数: {df['cites'].sum()}")
        print(f"平均每篇论文引用数: {df['cites_per_paper'].mean():.2f}")
        print(f"总高被引论文数: {df['top_papers'].sum()}")
        
        # 按引用次数排序
        print(f"\n按引用次数排序:")
        sorted_df = df.sort_values('cites', ascending=False)
        for _, row in sorted_df.iterrows():
            print(f"  {row['research_field']}: {row['cites']} 次引用")


def main():
    """主函数"""
    analyzer = ECNUDataAnalyzer("../downloads")
    
    # 分析所有文件
    results = analyzer.analyze_all_files()
    
    # 打印摘要
    analyzer.print_summary()
    
    # 保存结果
    analyzer.save_results_to_csv("ecnu_analysis_results.csv")
    
    return results


if __name__ == "__main__":
    main()