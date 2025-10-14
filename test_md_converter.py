#!/usr/bin/env python3
"""
測試 Markdown 轉換器
使用實際的檢測結果進行轉換測試
"""

import sys
from pathlib import Path

# 添加路徑
sys.path.append('.')
sys.path.append('src')

from src.lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector
from src.lchunk.converters.md_converter import MarkdownConverter

def test_md_converter():
    """測試 Markdown 轉換器"""
    print("🧪 測試 Markdown 轉換器")
    print("=" * 60)
    
    # 初始化檢測器
    try:
        detector = IntelligentHybridDetector()
        print("✅ 檢測器初始化成功")
    except Exception as e:
        print(f"❌ 檢測器初始化失敗: {e}")
        return
    
    # 測試檔案
    test_file = Path("data/samples/TPDM,111,侵訴,89,20250115,1.json")
    
    if not test_file.exists():
        print(f"❌ 測試檔案不存在: {test_file}")
        return
    
    print(f"📂 處理檔案: {test_file.name}")
    
    try:
        # 執行智能檢測
        result = detector.process_single_file(test_file)
        
        if not result or not result.line_based_chunks:
            print("❌ 沒有可用的分塊結果")
            return
        
        print(f"✅ 檢測完成，共 {len(result.line_based_chunks)} 個分塊")
        
        # 初始化 Markdown 轉換器
        converter = MarkdownConverter()
        
        # 轉換為 Markdown
        markdown_content = converter.convert_detection_result_to_markdown(
            result, include_metadata=True
        )
        
        print("\n📝 Markdown 轉換結果:")
        print("=" * 80)
        print(markdown_content)
        print("=" * 80)
        
        print(f"\n📊 轉換統計: {converter.conversion_stats}")
        
        # 保存結果
        output_path = Path("output/markdown")
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{test_file.stem}_converted.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"💾 結果已保存至: {output_file}")
        
    except Exception as e:
        print(f"❌ 轉換過程出錯: {e}")
        import traceback
        traceback.print_exc()

def test_batch_conversion():
    """測試批量轉換"""
    print("\n🔄 測試批量轉換")
    print("=" * 60)
    
    samples_dir = Path("data/samples")
    
    if not samples_dir.exists():
        print(f"❌ 樣本目錄不存在: {samples_dir}")
        return
    
    json_files = list(samples_dir.glob("*.json"))
    if not json_files:
        print("❌ 沒有找到 JSON 檔案")
        return
    
    print(f"📂 找到 {len(json_files)} 個 JSON 檔案")
    
    try:
        # 初始化
        detector = IntelligentHybridDetector()
        converter = MarkdownConverter()
        
        detection_results = []
        successful_conversions = 0
        
        # 處理每個檔案
        for json_file in json_files[:2]:  # 限制處理前2個檔案，避免記憶體問題
            print(f"\n🔍 處理: {json_file.name}")
            
            try:
                result = detector.process_single_file(json_file)
                if result and result.line_based_chunks:
                    detection_results.append(result)
                    print(f"✅ 檢測成功: {len(result.line_based_chunks)} 個分塊")
                else:
                    print("⚠️ 無有效分塊結果")
                    
            except Exception as e:
                print(f"❌ 檢測失敗: {e}")
        
        if detection_results:
            # 批量轉換
            output_dir = "output/markdown/batch"
            converted_files = converter.batch_convert_to_markdown(
                detection_results, output_dir=output_dir
            )
            
            print(f"\n✅ 批量轉換完成")
            print(f"📁 輸出目錄: {output_dir}")
            print(f"📊 成功轉換: {len(converted_files)} 個檔案")
            
            # 顯示轉換檔案列表
            for filename in converted_files.keys():
                print(f"   📄 {filename}")
        
    except Exception as e:
        print(f"❌ 批量轉換失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 執行測試
    test_md_converter()
    test_batch_conversion()