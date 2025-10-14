#!/usr/bin/env python3
"""
基於行的自適應層級符號檢測器演示腳本
Run Line-Based Adaptive Detector
"""

import sys
from pathlib import Path

# 添加專案路徑
sys.path.append('.')
sys.path.append('src')

from src.lchunk.detectors.adaptive_hybrid import IntelligentHybridDetector

def main():
    """主函數 - 基於行的自適應檢測演示"""
    print("🧠 基於行的自適應混合層級符號檢測器")
    print("新功能：特殊標記檢測 + 基於行的分層 + 內容合併")
    print("="*80)
    
    print("🎯 檢測邏輯：")
    print("  1. 檢測特殊標記：主文(L0) 理由(L0) 事實(L0) 事實及理由(L0) 日期(L-2)")
    print("  2. 主文前的行 = Header (L-3)")
    print("  3. 最後日期後的行 = Footer (L-3)")
    print("  4. 特殊標記間的非符號行 = Content (L-1)")
    print("  5. 層級符號行根據學習規則分配 L1, L2, L3, L4...")
    print("  6. 將相鄰的 L-1 內容合併到對應的層級符號下")
    print("")
    
    # 初始化自適應檢測器 (如果有BERT模型會自動載入)
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)

    # 處理 samples 目錄 (優先) 或 filtered 目錄
    sample_dirs = [
        Path("data/samples"),
        Path("data/processed/filtered")
    ]
    
    target_dir = None
    for sample_dir in sample_dirs:
        if sample_dir.exists() and list(sample_dir.glob("*.json")):
            target_dir = sample_dir
            break
    
    if target_dir:
        print(f"📁 處理目錄: {target_dir}")
        detector.process_sample_directory(target_dir)
    else:
        print("❌ 找不到包含JSON檔案的測試目錄")
        print("   請確保 data/samples/ 或 data/processed/filtered/ 目錄存在且包含JSON檔案")

if __name__ == "__main__":
    main()