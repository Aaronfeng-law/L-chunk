#!/usr/bin/env python3
"""
基於行的自適應層級檢測演示
展示完整的 line-based approach 流程：

1. 檢測特殊標記：主文(lv 0)、理由(lv 0)、事實(lv 0)、事實及理由(lv 0)、日期(lv -2)
2. Header/Footer 區域：主文之前(lv -3)、最後日期之後(lv -3)  
3. 內容區域：兩個層級符號行之間(lv -1)
4. 層級符號檢測：基於 R-D 或 S-D 學習的規則(lv 1,2,3,4...)
5. 相同層級內容合併：將所有 Lv -1 內容合併等
"""

import sys
import json
from pathlib import Path

# 添加 src 路徑
sys.path.append('src')

from lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector

def demonstrate_line_based_approach():
    """演示基於行的分塊方法"""
    print("🚀 基於行的自適應層級檢測演示")
    print("="*80)
    print("流程: 檢測特殊標記 → 學習層級規則 → 基於行分塊 → 內容合併")
    print()
    
    # 初始化檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)
    
    # 選擇測試檔案
    sample_dir = Path("data/samples")
    test_files = list(sample_dir.glob("*.json"))
    
    if not test_files:
        print("❌ 沒有找到測試檔案")
        return
    
    test_file = test_files[0]
    print(f"📁 處理檔案: {test_file.name}")
    
    # 執行完整處理流程
    result = detector.process_single_file(test_file)
    
    if not result:
        print("❌ 處理失敗")
        return
    
    print(f"\n✅ 處理完成！")
    
    # === 步驟1: 特殊標記檢測結果 ===
    print(f"\n📍 步驟1: 特殊標記檢測")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = data['JFULL'].split('\n')
    special_markers = detector.detect_special_markers(lines)
    
    for marker_type, line_numbers in special_markers.items():
        if line_numbers:
            print(f"  ✓ {marker_type}: 第 {[n+1 for n in line_numbers]} 行")
        else:
            print(f"  ✗ {marker_type}: 未檢測到")
    
    # === 步驟2: 學習層級規則 ===
    print(f"\n🎓 步驟2: 從 {result.learning_region} 學習層級規則")
    for rule in result.learned_rules:
        print(f"  📋 {rule.symbol_category}: Level {rule.assigned_level} (信心度: {rule.confidence:.3f})")
    
    # === 步驟3: 基於行的分塊統計 ===
    print(f"\n🏗️ 步驟3: 基於行分塊結果")
    if result.line_based_chunks:
        level_stats = {}
        for chunk in result.line_based_chunks:
            level = chunk.level
            chunk_type = chunk.chunk_type
            key = f"Lv{level}_{chunk_type}"
            level_stats[key] = level_stats.get(key, 0) + 1
        
        for key, count in sorted(level_stats.items()):
            print(f"  📦 {key}: {count} 個分塊")
    
    # === 步驟4: 層級內容合併 ===
    print(f"\n🔗 步驟4: 層級內容合併統計")
    stats = result.processing_stats
    if 'level_content_summary' in stats:
        for level, line_count in sorted(stats['level_content_summary'].items()):
            print(f"  📄 {level}: {line_count} 行內容")
    
    # === 步驟5: 分塊示例 ===
    print(f"\n📋 步驟5: 各層級分塊示例")
    
    if result.line_based_chunks:
        # 按層級分組
        level_groups = {}
        for chunk in result.line_based_chunks:
            level = chunk.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(chunk)
        
        # 顯示每個層級的代表性示例
        for level in sorted(level_groups.keys()):
            chunks = level_groups[level]
            print(f"\n  🎯 Level {level} 示例:")
            
            # 根據層級類型選擇顯示邏輯
            if level == -3:
                print(f"    Header/Footer 區域: {len(chunks)} 個分塊")
                if chunks:
                    for chunk in chunks[:2]:
                        content_preview = chunk.content_lines[0][:50] + "..." if chunk.content_lines else ""
                        print(f"    📝 {chunk.chunk_type}: 行{chunk.start_line+1}-{chunk.end_line+1}")
                        print(f"       {content_preview}")
            
            elif level == -2:
                print(f"    日期標記: {len(chunks)} 個")
                for chunk in chunks:
                    content = chunk.content_lines[0] if chunk.content_lines else ""
                    print(f"    📅 行{chunk.start_line+1}: {content.strip()}")
            
            elif level == -1:
                print(f"    內容區域: {len(chunks)} 個分塊")
                for chunk in chunks[:3]:  # 只顯示前3個
                    content_preview = chunk.content_lines[0][:40] + "..." if chunk.content_lines else ""
                    print(f"    📄 行{chunk.start_line+1}-{chunk.end_line+1}: {content_preview}")
                if len(chunks) > 3:
                    print(f"    ... 還有 {len(chunks) - 3} 個內容分塊")
            
            elif level == 0:
                print(f"    特殊標記: {len(chunks)} 個")
                for chunk in chunks:
                    content = chunk.content_lines[0] if chunk.content_lines else ""
                    print(f"    🏷️ {chunk.chunk_type}: {content.strip()}")
            
            else:  # level >= 1
                print(f"    層級符號: {len(chunks)} 個")
                for chunk in chunks[:3]:  # 只顯示前3個
                    symbol = chunk.leveling_symbol or ""
                    content = chunk.content_lines[0] if chunk.content_lines else ""
                    content_preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"    📌 [{symbol}] 行{chunk.start_line+1}: {content_preview}")
                if len(chunks) > 3:
                    print(f"    ... 還有 {len(chunks) - 3} 個層級符號")
    
    print(f"\n✅ 基於行的自適應層級檢測演示完成！")
    print(f"   總共處理了 {stats['total_lines']} 行")
    print(f"   生成了 {stats.get('line_based_chunks_count', 0)} 個分塊")
    print(f"   覆蓋了 {len(stats.get('level_content_summary', {}))} 個層級")

if __name__ == "__main__":
    demonstrate_line_based_approach()