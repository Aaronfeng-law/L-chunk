#!/usr/bin/env python3
"""
法院映射生成工具
從原始的 court_mapping_grouped.json 生成三種不同用途的映射文件
"""

import json
import os
import argparse
from typing import Dict, Any


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='生成法院映射文件')
    parser.add_argument('--input', '-i', default='court_mapping_grouped.json',
                       help='輸入的法院映射文件（默認：court_mapping_grouped.json）')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='輸出目錄（默認：當前目錄）')
    parser.add_argument('--split', action='store_true',
                       help='分離映射文件為三個不同用途的文件')
    
    args = parser.parse_args()
    
    if args.split:
        print("正在分離映射文件...")
        from split_mappings import main as split_main
        split_main()
    else:
        print("請使用 --split 參數來分離映射文件")
        print("或者直接運行: python split_mappings.py")


if __name__ == '__main__':
    main()