#!/usr/bin/env python3
"""
測試基於行的自適應層級檢測器
"""

import sys
from pathlib import Path

# 添加專案路徑
sys.path.append('.')
sys.path.append('src')

from src.lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector

def test_line_based_detection():
    """測試基於行的檢測"""
    print("🧪 測試基於行的自適應層級檢測")
    print("="*60)
    
    # 初始化檢測器 (不需要BERT模型也能工作)
    detector = IntelligentHybridDetector()
    
    # 測試樣本檔案
    sample_dir = Path("data/samples")
    if not sample_dir.exists():
        # 回退到 filtered 目錄
        sample_dir = Path("data/processed/filtered")
    
    if not sample_dir.exists():
        print("❌ 找不到測試數據目錄")
        return
    
    # 找到第一個JSON檔案進行測試
    json_files = list(sample_dir.glob("*.json"))
    if not json_files:
        print("❌ 找不到JSON測試檔案")
        return
    
    test_file = json_files[0]
    print(f"📄 測試檔案: {test_file.name}")
    
    # 執行基於行的檢測
    result = detector.process_single_file(test_file)
    
    if result:
        print(f"\n✅ 檢測成功!")
        print(f"   檔案: {result.filename}")
        print(f"   學習模式: {result.learning_region}")
        print(f"   處理統計:")
        
        stats = result.processing_stats
        for key, value in stats.items():
            print(f"     {key}: {value}")
        
        # 顯示基於行的分塊結果
        if 'line_based_chunks' in result.applied_hierarchy:
            chunks = result.applied_hierarchy['line_based_chunks']
            print(f"\n📋 基於行的分塊結果:")
            
            for level in sorted(chunks.keys()):
                chunk_count = len(chunks[level])
                level_name = {
                    -3: "Header/Footer", 
                    -2: "日期", 
                    -1: "內容", 
                    0: "特殊標記"
                }.get(level, f"層級符號")
                
                print(f"   L{level} ({level_name}): {chunk_count} 個分塊")
                
                # 顯示前2個分塊的詳細內容
                for i, chunk in enumerate(chunks[level][:2]):
                    print(f"     分塊 {i+1}:")
                    
                    if 'symbol_lines' in chunk and chunk['symbol_lines']:
                        print(f"       符號行: {len(chunk['symbol_lines'])} 行")
                        for sym_line in chunk['symbol_lines'][:2]:
                            print(f"         行{sym_line['line_number']:4}: {sym_line['line_text'][:50]}...")
                    
                    if 'content_lines' in chunk and chunk['content_lines']:
                        print(f"       內容行: {len(chunk['content_lines'])} 行")
                        for content_line in chunk['content_lines'][:2]:
                            print(f"         行{content_line['line_number']:4}: {content_line['line_text'][:50]}...")
                    
                    if 'lines' in chunk and chunk['lines']:
                        print(f"       總行數: {len(chunk['lines'])} 行")
                        for line_item in chunk['lines'][:2]:
                            print(f"         行{line_item['line_number']:4}: {line_item['line_text'][:50]}...")
        
        print(f"\n🎯 測試完成!")
    else:
        print("❌ 檢測失敗")

def test_special_markers():
    """測試特殊標記檢測"""
    print("\n🔍 測試特殊標記檢測")
    print("-"*40)
    
    # 創建測試行
    test_lines = [
        "這是標題行",
        "主文",
        "一、第一層",
        "(一)第二層",
        "這是內容行1",
        "這是內容行2", 
        "事實",
        "二、事實內容",
        "理由",
        "三、理由內容",
        "事實及理由",
        "四、合併內容",
        "中華民國113年10月14日",
        "這是最後一行"
    ]
    
    detector = IntelligentHybridDetector()
    
    # 檢測特殊標記
    special_markers = detector.detect_special_markers(test_lines)
    
    print("檢測到的特殊標記:")
    for marker_type, markers in special_markers.items():
        if markers:
            print(f"  {marker_type}:")
            for line_num, line_text in markers:
                print(f"    行{line_num:2}: {line_text}")
    
    # 測試基於行的層級分析
    line_hierarchy = detector.create_line_based_hierarchy(test_lines)
    
    print(f"\n行級別映射:")
    line_levels = line_hierarchy['line_levels']
    for line_num in sorted(line_levels.keys()):
        level = line_levels[line_num]
        line_text = test_lines[line_num] if line_num < len(test_lines) else ""
        level_name = {
            -3: "Header/Footer", 
            -2: "日期", 
            -1: "內容", 
            0: "特殊標記", 
            None: "待分配"
        }.get(level, f"L{level}")
        
        print(f"  行{line_num:2} (L{level if level is not None else '?'} {level_name}): {line_text}")

if __name__ == "__main__":
    test_special_markers()
    test_line_based_detection()