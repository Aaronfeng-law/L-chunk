#!/usr/bin/env python3
"""檢查 RAG chunks 中 location 欄位的統計資訊"""

import json
import sys
from collections import Counter
from pathlib import Path


def check_location_stats(json_file: str):
    """檢查 JSON 文件中 location 欄位的統計"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rag_chunks = data.get('rag_chunks', [])
    
    print(f"📊 Location 統計報告")
    print(f"{'='*60}")
    print(f"檔案: {Path(json_file).name}")
    print(f"總 chunks 數: {len(rag_chunks)}")
    print()
    
    # 統計 location
    location_counter = Counter()
    location_types = {}  # location -> [chunk_types]
    
    for chunk in rag_chunks:
        location = chunk.get('location', 'MISSING')
        chunk_type = chunk.get('chunk_type', 'unknown')
        
        location_counter[location] += 1
        
        if location not in location_types:
            location_types[location] = set()
        location_types[location].add(chunk_type)
    
    # 顯示統計結果
    print("📍 Location 分佈:")
    print(f"{'-'*60}")
    
    location_names = {
        'H': 'Header (標頭)',
        'M': 'Main (主文)',
        'F': 'Facts (事實)',
        'R': 'Reasons (理由)',
        'S': 'Facts and Reasons (事實及理由)',
        'D1': 'Date1 (第一個日期)',
        'D2': 'Date2 (第二個日期)',
        'SIG': 'Signature (署名區)',
        'A': 'Appendix (附錄)',
        'O': 'Other (其他)',
        'MISSING': '⚠️  缺少 location 欄位'
    }
    
    for location, count in sorted(location_counter.items(), key=lambda x: -x[1]):
        name = location_names.get(location, location)
        percentage = (count / len(rag_chunks)) * 100
        types = ', '.join(sorted(location_types[location]))
        
        print(f"  {location:6s} ({name:30s}): {count:3d} ({percentage:5.1f}%)")
        print(f"         相關 chunk_types: {types}")
    
    print()
    print(f"{'='*60}")
    
    # 檢查是否有缺少 location 的 chunks
    missing_count = location_counter.get('MISSING', 0)
    if missing_count > 0:
        print(f"⚠️  警告: 有 {missing_count} 個 chunks 缺少 location 欄位!")
        print()
        print("缺少 location 的 chunks:")
        for i, chunk in enumerate(rag_chunks[:10]):  # 只顯示前 10 個
            if 'location' not in chunk:
                print(f"  - {chunk.get('chunk_id', 'N/A')}: {chunk.get('chunk_type', 'N/A')}")
    else:
        print("✅ 所有 chunks 都包含 location 欄位")
    
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_location_stats.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"❌ 錯誤: 檔案不存在: {json_file}")
        sys.exit(1)
    
    check_location_stats(json_file)
