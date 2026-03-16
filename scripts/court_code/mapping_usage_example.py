#!/usr/bin/env python3
"""
使用三種不同映射文件的示例代碼
演示如何使用 code_to_fullcourt.json、code_to_base_court.json 和 code_to_sub_court.json
"""

import json
from typing import Dict, Any, Optional


class CourtMappingManager:
    """法院映射管理器，提供三種不同層級的查詢功能"""
    
    def __init__(self, mappings_dir: str = "."):
        """
        初始化映射管理器
        
        Args:
            mappings_dir: 映射文件所在目錄
        """
        self.mappings_dir = mappings_dir
        
        # 載入三種映射文件
        self.full_court_mapping = self._load_mapping("code_to_fullcourt.json")
        self.base_court_mapping = self._load_mapping("code_to_base_court.json")
        self.sub_court_mapping = self._load_mapping("code_to_sub_court.json")
        
        print(f"✓ 載入完整法院映射: {len(self.full_court_mapping)} 項目")
        print(f"✓ 載入基礎法院映射: {len(self.base_court_mapping)} 項目")
        print(f"✓ 載入子法院映射: {len(self.sub_court_mapping)} 項目")
    
    def _load_mapping(self, filename: str) -> Dict[str, Any]:
        """載入映射文件"""
        try:
            with open(f"{self.mappings_dir}/{filename}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('mapping', {})
        except FileNotFoundError:
            print(f"警告：找不到文件 {filename}")
            return {}
    
    def get_full_court_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        獲取完整的法院資訊
        
        Args:
            code: 法院代碼（基礎代碼或子代碼）
            
        Returns:
            完整的法院資訊字典，如果找不到則返回 None
        """
        return self.full_court_mapping.get(code)
    
    def get_base_court_name(self, code: str) -> Optional[str]:
        """
        快速獲取基礎法院名稱
        
        Args:
            code: 基礎法院代碼
            
        Returns:
            法院名稱，如果找不到則返回 None
        """
        return self.base_court_mapping.get(code)
    
    def get_sub_court_name(self, code: str) -> Optional[str]:
        """
        快速獲取子法院完整名稱
        
        Args:
            code: 子法院代碼
            
        Returns:
            完整法院名稱，如果找不到則返回 None
        """
        return self.sub_court_mapping.get(code)
    
    def search_courts_by_name(self, keyword: str) -> Dict[str, list]:
        """
        根據法院名稱關鍵字搜尋相關法院
        
        Args:
            keyword: 搜尋關鍵字
            
        Returns:
            包含匹配結果的字典
        """
        results = {
            'base_courts': [],
            'sub_courts': []
        }
        
        # 搜尋基礎法院
        for code, name in self.base_court_mapping.items():
            if keyword in name:
                results['base_courts'].append({
                    'code': code,
                    'name': name
                })
        
        # 搜尋子法院
        for code, name in self.sub_court_mapping.items():
            if keyword in name:
                results['sub_courts'].append({
                    'code': code,
                    'name': name
                })
        
        return results
    
    def get_court_hierarchy(self, code: str) -> Optional[Dict[str, Any]]:
        """
        獲取法院的層級資訊
        
        Args:
            code: 法院代碼
            
        Returns:
            包含層級資訊的字典
        """
        full_info = self.get_full_court_info(code)
        if not full_info:
            return None
        
        return {
            'code': code,
            'court_name': full_info.get('court_name'),
            'jurisdiction': full_info.get('jurisdiction'),
            'is_base_court': 'case_type' not in full_info,
            'case_type': full_info.get('case_type'),
            'base_court_code': full_info.get('base_court_code')
        }


def demo_usage():
    """演示如何使用映射管理器"""
    print("=" * 60)
    print("法院映射系統使用示例")
    print("=" * 60)
    
    # 初始化管理器
    manager = CourtMappingManager()
    
    print("\n" + "=" * 60)
    print("1. 基礎法院代碼查詢示例")
    print("=" * 60)
    
    base_codes = ["SJE", "TPS", "IPC"]
    for code in base_codes:
        name = manager.get_base_court_name(code)
        if name:
            print(f"基礎代碼 {code} -> {name}")
        else:
            print(f"找不到基礎代碼: {code}")
    
    print("\n" + "=" * 60)
    print("2. 子法院代碼查詢示例")
    print("=" * 60)
    
    sub_codes = ["SJEM", "TPSV", "IPCM"]
    for code in sub_codes:
        name = manager.get_sub_court_name(code)
        if name:
            print(f"子代碼 {code} -> {name}")
        else:
            print(f"找不到子代碼: {code}")
    
    print("\n" + "=" * 60)
    print("3. 完整法院資訊查詢示例")
    print("=" * 60)
    
    full_codes = ["SJE", "SJEM"]
    for code in full_codes:
        info = manager.get_full_court_info(code)
        if info:
            print(f"\n代碼 {code} 的完整資訊:")
            print(f"  法院名稱: {info.get('court_name')}")
            print(f"  法院層級: {info.get('jurisdiction')}")
            if 'case_type' in info:
                print(f"  案件類型: {info.get('case_type')}")
                print(f"  完整名稱: {info.get('full_name')}")
                print(f"  基礎代碼: {info.get('base_court_code')}")
            else:
                print(f"  案件類型: {list(info.get('case_types', {}).keys())}")
        else:
            print(f"找不到代碼: {code}")
    
    print("\n" + "=" * 60)
    print("4. 法院名稱搜尋示例")
    print("=" * 60)
    
    keywords = ["三重", "最高", "簡易"]
    for keyword in keywords:
        results = manager.search_courts_by_name(keyword)
        print(f"\n搜尋關鍵字: '{keyword}'")
        
        if results['base_courts']:
            print("  基礎法院:")
            for court in results['base_courts']:
                print(f"    {court['code']} -> {court['name']}")
        
        if results['sub_courts']:
            print("  子法院:")
            for court in results['sub_courts'][:3]:  # 只顯示前3個
                print(f"    {court['code']} -> {court['name']}")
            if len(results['sub_courts']) > 3:
                print(f"    ... 還有 {len(results['sub_courts']) - 3} 個結果")
    
    print("\n" + "=" * 60)
    print("5. 法院層級資訊查詢示例")
    print("=" * 60)
    
    hierarchy_codes = ["SJE", "SJEM", "TPS", "TPSV"]
    for code in hierarchy_codes:
        hierarchy = manager.get_court_hierarchy(code)
        if hierarchy:
            print(f"\n代碼 {code} 的層級資訊:")
            print(f"  法院名稱: {hierarchy['court_name']}")
            print(f"  法院層級: {hierarchy['jurisdiction']}")
            print(f"  是否為基礎法院: {hierarchy['is_base_court']}")
            if hierarchy['case_type']:
                print(f"  案件類型: {hierarchy['case_type']}")
            if hierarchy['base_court_code'] != code:
                print(f"  基礎法院代碼: {hierarchy['base_court_code']}")
        else:
            print(f"找不到代碼: {code}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    demo_usage()