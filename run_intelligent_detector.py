#!/usr/bin/env python3
"""
自適應檢測器運行腳本
"""

import sys
from pathlib import Path

# 添加 src 到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """運行自適應檢測器"""
    print("🧠 啟動自適應混合檢測器")
    print("基於重組後的項目結構")
    print("="*50)
    
    try:
        from lchunk.detectors.intelligent_hybrid import IntelligentHybridDetector
        
        # 檢查模型路徑
        model_path = "models/bert/level_detector/best_model"
        model_exists = Path(model_path).exists()
        
        print(f"📦 BERT 模型: {'存在' if model_exists else '不存在'}")
        
        # 初始化檢測器
        detector = IntelligentHybridDetector(model_path if model_exists else None)
        
        # 檢查樣本數據
        sample_dir = Path("data/samples")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.json"))
            print(f"📊 找到 {len(sample_files)} 個樣本文件")
            
            if sample_files:
                # 處理第一個樣本文件
                test_file = sample_files[0]
                print(f"\n🔍 測試檢測: {test_file.name}")
                
                result = detector.process_single_file(test_file)
                if result:
                    print(f"✅ 檢測成功")
                    print(f"   學習區間: {result.learning_region}")
                    print(f"   學習規則數: {len(result.learned_rules)}")
                    print(f"   檢測符號數: {len([r for r in result.full_detection_results if r.final_prediction])}")
                else:
                    print("❌ 檢測失敗")
        else:
            print("⚠️ 未找到樣本數據目錄")
            
    except Exception as e:
        print(f"❌ 運行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()