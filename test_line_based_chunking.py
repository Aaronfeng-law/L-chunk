#!/usr/bin/env python3
"""
測試基於行的分塊功能
驗證新的 line-based approach 是否按照要求工作：
1. 檢測特殊標記 (主文、理由、事實、事實及理由、日期)
2. Header/Footer 區域劃分 (Lv -3)
3. 內容區域分割 (Lv -1)
4. 層級符號檢測與分配 (Lv 1,2,3...)
5. 相同層級內容合併
"""

import sys
import json
from pathlib import Path

# 添加 src 路徑
sys.path.append('src')

from lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector

def test_line_based_chunking():
    """測試基於行的分塊功能"""
    print("🧪 測試基於行的分塊功能")
    print("="*80)
    
    # 選擇測試檔案
    sample_dir = Path("data/samples")
    test_files = list(sample_dir.glob("*.json"))
    
    if not test_files:
        print("❌ 沒有找到測試檔案")
        return
    
    # 選擇第一個檔案進行詳細測試
    test_file = test_files[0]
    print(f"📁 測試檔案: {test_file.name}")
    
    # 初始化檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)
    
    # 處理檔案
    result = detector.process_single_file(test_file)
    
    if not result:
        print("❌ 檔案處理失敗")
        return
    
    print(f"\n✅ 檔案處理成功")
    print(f"📊 學習區間: {result.learning_region}")
    print(f"🔢 學習規則數: {len(result.learned_rules)}")
    
    # 檢查基於行的分塊結果
    if result.line_based_chunks:
        print(f"\n🏗️ 基於行的分塊結果: {len(result.line_based_chunks)} 個分塊")
        
        # 按層級分組顯示
        level_groups = {}
        for chunk in result.line_based_chunks:
            level = chunk.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(chunk)
        
        print("\n📋 各層級分塊詳情:")
        for level in sorted(level_groups.keys()):
            chunks = level_groups[level]
            print(f"\n  🎯 Level {level}: {len(chunks)} 個分塊")
            
            for i, chunk in enumerate(chunks[:3]):  # 只顯示前3個
                content_preview = chunk.content_lines[0][:50] + "..." if chunk.content_lines else ""
                symbol_info = f" [{chunk.leveling_symbol}]" if chunk.leveling_symbol else ""
                print(f"    {i+1}. {chunk.chunk_type}{symbol_info}: 行{chunk.start_line+1}-{chunk.end_line+1}")
                print(f"       內容: {content_preview}")
            
            if len(chunks) > 3:
                print(f"    ... 還有 {len(chunks) - 3} 個分塊")
        
        # 顯示處理統計
        stats = result.processing_stats
        print(f"\n📈 處理統計:")
        print(f"  - 總行數: {stats['total_lines']}")
        print(f"  - 學習行數: {stats['learning_lines']}")
        print(f"  - 檢測符號數: {stats['total_symbols_detected']}")
        print(f"  - 學習規則數: {stats['learned_rules_count']}")
        print(f"  - 基於行分塊數: {stats.get('line_based_chunks_count', 0)}")
        
        if 'level_content_summary' in stats:
            print(f"\n📊 層級內容統計:")
            for level, line_count in sorted(stats['level_content_summary'].items()):
                print(f"  - {level}: {line_count} 行")
    
    else:
        print("❌ 未生成基於行的分塊結果")
    
    # 測試特殊標記檢測
    print(f"\n🔍 測試特殊標記檢測...")
    
    # 讀取檔案內容
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = data['JFULL'].split('\n')
    special_markers = detector.detect_special_markers(lines)
    
    print(f"🎯 特殊標記檢測結果:")
    for marker_type, line_numbers in special_markers.items():
        if line_numbers:
            print(f"  - {marker_type}: {len(line_numbers)} 個")
            for line_num in line_numbers[:3]:  # 只顯示前3個
                print(f"    行 {line_num+1}: {lines[line_num].strip()}")
            if len(line_numbers) > 3:
                print(f"    ... 還有 {len(line_numbers) - 3} 個")
        else:
            print(f"  - {marker_type}: 未找到")

def test_multiple_files():
    """測試多個檔案的批量處理"""
    print(f"\n🚀 測試批量處理...")
    
    # 初始化檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)
    
    # 處理 samples 目錄
    sample_dir = Path("data/samples") 
    if sample_dir.exists():
        detector.process_sample_directory(sample_dir)
    else:
        print(f"❌ 目錄不存在: {sample_dir}")

if __name__ == "__main__":
    print("🧪 基於行的分塊功能測試")
    print("="*80)
    
    # 測試單個檔案的詳細分塊
    test_line_based_chunking()
    
    # 測試批量處理
    # test_multiple_files()
    
    print("\n✅ 測試完成")