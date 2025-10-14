#!/usr/bin/env python3
"""
測試所有 samples 文件的 line-based chunking
整合修正後的特殊標記檢測功能
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from src.lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector

def test_all_samples():
    """測試所有 samples 文件"""
    print("🚀 測試所有 samples 文件 - Line-Based Chunking")
    print("="*80)
    
    # 初始化檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)
    
    # samples 目錄
    samples_dir = Path("data/samples")
    
    if not samples_dir.exists():
        print(f"❌ 目錄不存在: {samples_dir}")
        return
    
    # 獲取所有 JSON 文件
    json_files = list(samples_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 在 {samples_dir} 中沒有找到 JSON 檔案")
        return
    
    print(f"📁 找到 {len(json_files)} 個檔案:")
    for file in json_files:
        print(f"   - {file.name}")
    print()
    
    # 處理每個檔案
    all_results = []
    summary_stats = {
        'total_files': len(json_files),
        'successful_files': 0,
        'failed_files': 0,
        'total_chunks': 0,
        'total_lines': 0,
        'special_markers_found': 0,
        'learning_regions': {'S-D': 0, 'R-D': 0, '全文': 0}
    }
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] 處理: {json_file.name}")
        print("-" * 60)
        
        try:
            result = detector.process_single_file(json_file)
            
            if result and result.line_based_chunks:
                all_results.append(result)
                summary_stats['successful_files'] += 1
                summary_stats['total_chunks'] += len(result.line_based_chunks)
                summary_stats['total_lines'] += result.processing_stats['total_lines']
                summary_stats['learning_regions'][result.learning_region] += 1
                
                # 計算特殊標記數量
                special_chunks = [c for c in result.line_based_chunks 
                                if c.chunk_type in ['main_text', 'facts', 'reasons', 'facts_and_reasons']]
                summary_stats['special_markers_found'] += len(special_chunks)
                
                # 顯示檔案結果
                print(f"✅ 成功處理")
                print(f"   📊 學習模式: {result.learning_region}")
                print(f"   📝 總行數: {result.processing_stats['total_lines']:,}")
                print(f"   🧩 生成分塊: {len(result.line_based_chunks)}")
                print(f"   🎯 特殊標記: {len(special_chunks)} 個")
                
                if special_chunks:
                    print("   📍 檢測到的特殊標記:")
                    for chunk in special_chunks:
                        content = chunk.content_lines[0] if chunk.content_lines else ''
                        print(f"      - {chunk.chunk_type}: 行 {chunk.start_line + 1} 「{content.strip()}」")
                
                # 層級統計
                level_stats = {}
                for chunk in result.line_based_chunks:
                    level = chunk.level
                    level_stats[level] = level_stats.get(level, 0) + 1
                
                print("   📊 層級分布:")
                for level in sorted(level_stats.keys()):
                    print(f"      Lv {level}: {level_stats[level]} 個分塊")
                
            else:
                print(f"❌ 處理失敗")
                summary_stats['failed_files'] += 1
                
        except Exception as e:
            print(f"❌ 處理錯誤: {e}")
            summary_stats['failed_files'] += 1
    
    # 生成總結報告
    print("\n" + "="*80)
    print("📊 總結報告")
    print("="*80)
    
    print(f"📁 處理檔案: {summary_stats['total_files']} 個")
    print(f"✅ 成功: {summary_stats['successful_files']} 個")
    print(f"❌ 失敗: {summary_stats['failed_files']} 個")
    print(f"📝 總行數: {summary_stats['total_lines']:,} 行")
    print(f"🧩 總分塊數: {summary_stats['total_chunks']:,} 個")
    print(f"🎯 特殊標記總數: {summary_stats['special_markers_found']} 個")
    
    print(f"\n📊 學習區間分布:")
    for region, count in summary_stats['learning_regions'].items():
        if count > 0:
            print(f"   {region}: {count} 個檔案")
    
    if all_results:
        print(f"\n📈 平均統計:")
        avg_lines = summary_stats['total_lines'] / len(all_results)
        avg_chunks = summary_stats['total_chunks'] / len(all_results)
        avg_markers = summary_stats['special_markers_found'] / len(all_results)
        
        print(f"   平均行數: {avg_lines:.0f} 行/檔案")
        print(f"   平均分塊: {avg_chunks:.0f} 個/檔案")
        print(f"   平均特殊標記: {avg_markers:.1f} 個/檔案")
        
        # 詳細層級統計
        print(f"\n🔍 詳細分析:")
        
        # 合併所有檔案的層級統計
        combined_level_stats = {}
        for result in all_results:
            for chunk in result.line_based_chunks:
                level = chunk.level
                chunk_type = chunk.chunk_type
                key = f"Lv_{level}_{chunk_type}"
                combined_level_stats[key] = combined_level_stats.get(key, 0) + 1
        
        print("   跨檔案層級分布:")
        for key in sorted(combined_level_stats.keys()):
            print(f"      {key}: {combined_level_stats[key]} 個")

if __name__ == "__main__":
    test_all_samples()