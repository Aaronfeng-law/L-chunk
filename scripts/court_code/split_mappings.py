#!/usr/bin/env python3
"""
將 court_mapping_grouped.json 分離成三個不同的映射文件：
1. code_to_fullcourt.json - 完整的法院資訊映射
2. code_to_base_court.json - 基礎法院代碼到法院名稱的映射
3. code_to_sub_court.json - 子法院代碼到法院名稱的映射
"""

import json
import os
from typing import Dict, Any


def load_grouped_mapping(file_path: str) -> Dict[str, Any]:
    """載入分組的法院映射資料"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_full_court_mapping(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    創建完整的法院資訊映射 (code_to_fullcourt)
    包含所有詳細資訊的原始版本
    """
    full_mapping = {}
    courts = data.get('courts', {})
    
    for court_name, court_info in courts.items():
        base_code = court_info.get('base_court_code', '')
        
        # 為基礎法院代碼創建映射
        if base_code and base_code not in full_mapping:
            full_mapping[base_code] = {
                'court_name': court_info.get('court_name', ''),
                'jurisdiction': court_info.get('jurisdiction', ''),
                'base_court_code': base_code,
                'case_types': court_info.get('case_types', {})
            }
        
        # 為每個子法院代碼創建映射
        case_types = court_info.get('case_types', {})
        for case_type, case_info in case_types.items():
            sub_code = case_info.get('sub_court_code', '')
            if sub_code:
                full_mapping[sub_code] = {
                    'court_name': court_info.get('court_name', ''),
                    'jurisdiction': court_info.get('jurisdiction', ''),
                    'base_court_code': base_code,
                    'case_type': case_type,
                    'sub_court_code': sub_code,
                    'full_name': case_info.get('full_name', '')
                }
    
    return full_mapping


def create_base_court_mapping(data: Dict[str, Any]) -> Dict[str, str]:
    """
    創建基礎法院代碼映射 (code_to_base_court)
    簡單的基礎法院代碼到法院名稱的映射
    """
    base_mapping = {}
    courts = data.get('courts', {})
    
    for court_name, court_info in courts.items():
        base_code = court_info.get('base_court_code', '')
        court_name_clean = court_info.get('court_name', '')
        
        if base_code and court_name_clean:
            base_mapping[base_code] = court_name_clean
    
    return base_mapping


def create_sub_court_mapping(data: Dict[str, Any]) -> Dict[str, str]:
    """
    創建子法院代碼映射 (code_to_sub_court)
    簡單的子法院代碼到完整名稱的映射
    """
    sub_mapping = {}
    courts = data.get('courts', {})
    
    for court_name, court_info in courts.items():
        case_types = court_info.get('case_types', {})
        
        for case_type, case_info in case_types.items():
            sub_code = case_info.get('sub_court_code', '')
            full_name = case_info.get('full_name', '')
            
            if sub_code and full_name:
                sub_mapping[sub_code] = full_name
    
    return sub_mapping


def save_mapping(mapping: Dict, filename: str, description: str):
    """儲存映射到 JSON 文件"""
    output_data = {
        'metadata': {
            'description': description,
            'total_entries': len(mapping),
            'generated_from': 'court_mapping_grouped.json'
        },
        'mapping': mapping
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {filename}，包含 {len(mapping)} 個映射項目")


def main():
    """主函數"""
    # 設定文件路徑
    input_file = 'court_mapping_grouped.json'
    
    if not os.path.exists(input_file):
        print(f"錯誤：找不到輸入文件 {input_file}")
        return
    
    # 載入原始資料
    print(f"載入原始資料：{input_file}")
    data = load_grouped_mapping(input_file)
    
    # 創建三種不同的映射
    print("創建映射...")
    
    # 1. 完整法院資訊映射
    full_court_mapping = create_full_court_mapping(data)
    save_mapping(
        full_court_mapping,
        'code_to_fullcourt.json',
        '完整的法院資訊映射，包含所有詳細資料和案件類型資訊'
    )
    
    # 2. 基礎法院代碼映射
    base_court_mapping = create_base_court_mapping(data)
    save_mapping(
        base_court_mapping,
        'code_to_base_court.json',
        '基礎法院代碼到法院名稱的簡單映射，用於快速查詢'
    )
    
    # 3. 子法院代碼映射
    sub_court_mapping = create_sub_court_mapping(data)
    save_mapping(
        sub_court_mapping,
        'code_to_sub_court.json',
        '子法院代碼到完整法院名稱的簡單映射，用於快速查詢'
    )
    
    print("\n映射分離完成！")
    print(f"- code_to_fullcourt.json: {len(full_court_mapping)} 個項目")
    print(f"- code_to_base_court.json: {len(base_court_mapping)} 個項目")
    print(f"- code_to_sub_court.json: {len(sub_court_mapping)} 個項目")


if __name__ == '__main__':
    main()