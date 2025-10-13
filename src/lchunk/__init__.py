"""
L-chunk: 法律文檔層級符號檢測系統

基於 Linus "Good taste" 原則的實用主義設計：
- 簡單而有效的架構
- 清晰的模組分離
- 最小化複雜度
"""

__version__ = "0.1.0"
__author__ = "L-chunk Team"

# 主要檢測器
from .detectors.ultra_strict import UltraStrictDetector
from .detectors.hybrid import HybridLevelSymbolDetector
from .detectors.intelligent_hybrid import IntelligentHybridDetector

# 分析器
from .analyzers.comprehensive import analyze_filtered_dataset
from .analyzers.splitter import process_single_file

__all__ = [
    "UltraStrictDetector",
    "HybridLevelSymbolDetector", 
    "IntelligentHybridDetector",
    "analyze_filtered_dataset",
    "process_single_file",
]
