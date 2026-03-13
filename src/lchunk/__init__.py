"""
L-chunk: 法律文檔層級符號檢測系統

"""

__version__ = "0.1.0"
__author__ = "L-chunk Team"

# 主要檢測器
from .detectors.ultra_strict import UltraStrictDetector
from .detectors.hybrid import HybridLevelSymbolDetector
from .detectors.adaptive_hybrid import AdaptiveHybridDetector

# 分析器
from .analyzers.comprehensive import analyze_filtered_dataset
from .analyzers.splitter import process_single_file

__all__ = [
    "UltraStrictDetector",
    "HybridLevelSymbolDetector", 
    "AdaptiveHybridDetector",
    "analyze_filtered_dataset",
    "process_single_file",
]
