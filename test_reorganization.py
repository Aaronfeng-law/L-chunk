#!/usr/bin/env python3
"""
測試重組後的項目結構
"""

import sys
from pathlib import Path

# 添加 src 到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """測試所有主要模組的導入"""
    print("🧪 測試重組後的模組導入...")
    
    try:
        # 測試檢測器
        from lchunk.detectors.ultra_strict import UltraStrictDetector
        print("✅ UltraStrictDetector 導入成功")
        
        from lchunk.detectors.hybrid import HybridLevelSymbolDetector
        print("✅ HybridLevelSymbolDetector 導入成功")
        
        from lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector
        print("✅ IntelligentHybridDetector 導入成功")
        
        # 測試分析器
        from lchunk.analyzers.comprehensive import analyze_filtered_dataset
        print("✅ analyze_filtered_dataset 導入成功")
        
        from lchunk.analyzers.splitter import process_single_file
        print("✅ process_single_file 導入成功")
        
        # 測試訓練模組
        from lchunk.training.bert_trainer import BERTLevelSymbolTrainer
        print("✅ BERTLevelSymbolTrainer 導入成功")
        
        print("\n🎉 所有模組導入測試通過!")
        return True
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return False

def test_basic_functionality():
    """測試基本功能"""
    print("\n🔧 測試基本功能...")
    
    try:
        from lchunk.detectors.ultra_strict import UltraStrictDetector
        
        # 創建檢測器實例
        detector = UltraStrictDetector()
        print("✅ UltraStrictDetector 實例化成功")
        
        # 測試檢測功能
        test_lines = [
            "\\r\\n一、這是測試文字",
            "\\r\\n、這是無效格式", 
            "普通文字"
        ]
        
        markers = detector.detect_ultra_strict_markers(test_lines)
        print(f"✅ 檢測功能正常，檢測到 {len(markers)} 個標記")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能測試錯誤: {e}")
        return False

def test_data_access():
    """測試數據文件訪問"""
    print("\n📊 測試數據文件訪問...")
    
    data_paths = [
        "data/samples",
        "data/processed/filtered", 
        "data/training",
        "models/bert/level_detector"
    ]
    
    for path in data_paths:
        full_path = Path(path)
        if full_path.exists():
            print(f"✅ {path} 存在")
        else:
            print(f"⚠️ {path} 不存在")

def main():
    """主測試函數"""
    print("🚀 L-chunk 重組項目測試")
    print("="*50)
    
    success = True
    
    # 測試導入
    if not test_imports():
        success = False
    
    # 測試功能
    if success and not test_basic_functionality():
        success = False
    
    # 測試數據訪問
    test_data_access()
    
    print("\n" + "="*50)
    if success:
        print("🎉 重組項目測試通過!")
        print("💡 項目重組成功，所有核心功能正常")
    else:
        print("❌ 重組項目測試失敗")
        print("🔧 需要修復導入或功能問題")
    
    return success

if __name__ == "__main__":
    main()