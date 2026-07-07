#!/usr/bin/env python3
"""
简化版数据清洗脚本
"""

import pandas as pd
from pathlib import Path
import sys

def clean_all_data():
    """清理所有数据文件"""
    
    # 你的数据根目录
    data_root = Path("motion_data")
    
    if not data_root.exists():
        print(f"错误: 目录不存在: {data_root}")
        return
    
    # 查找所有CSV文件
    csv_files = []
    for ext in ['*.csv', '*.CSV']:
        csv_files.extend(data_root.rglob(ext))
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        print(f"处理: {csv_file}")
        
        try:
            # 读取文件
            df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
            
            # 清理列名
            df.columns = [str(col).strip() for col in df.columns]
            
            # 清理每个单元格的前导空格
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.lstrip()
            
            # 保存回原文件
            df.to_csv(csv_file, index=False)
            print(f"  ✓ 清理完成")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")

if __name__ == "__main__":
    print("开始清理数据中的前导空格...")
    print("警告: 这将直接修改原文件!")
    
    confirm = input("是否继续? (y/n): ")
    if confirm.lower() == 'y':
        clean_all_data()
        print("\n✅ 清理完成!")
    else:
        print("操作取消")