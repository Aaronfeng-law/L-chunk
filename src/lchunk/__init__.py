"""
L-chunk: 法律文檔層級符號檢測系統

"""

__version__ = "0.1.0"
__author__ = "L-chunk Team"

# Pipeline (資料結構 + 管線入口)
from .pipeline import (
    JudgmentArtifact,
    DocumentLine,
    SectionContent,
    SymbolDetection,
    PipelineOrchestrator,
)

# Splitter (Phase 1)
from .analyzers.splitter_refactor import (
    split_judgment_document,
    process_single_file as process_single_file_refactor,
    normalize_text,
    find_section_pattern,
    classify_document_sections,
)

# 檢測器
from .detectors.ultra_strict import UltraStrictDetector
from .detectors.soft_with_bert import HybridLevelSymbolDetector
from .detectors.adaptive_hierarchy import AdaptiveHybridDetector

# 分析器
# from .analyzers.comprehensive import analyze_filtered_dataset
from .analyzers.splitter_refactor import process_single_file

__all__ = [
    # Pipeline
    "JudgmentArtifact",
    "DocumentLine",
    "SectionContent",
    "SymbolDetection",
    "PipelineOrchestrator",
    # Splitter
    "split_judgment_document",
    "normalize_text",
    "find_section_pattern",
    "classify_document_sections",
    # Detectors
    "UltraStrictDetector",
    "HybridLevelSymbolDetector", 
    "AdaptiveHybridDetector",
    # Analyzers
    "analyze_filtered_dataset",
    "process_single_file",
]
