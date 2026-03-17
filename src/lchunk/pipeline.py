#!/usr/bin/env python3
"""
統一資料結構中心與整個處理管線 (Pipeline) 的進入點

資料結構定義：SymbolDetection, DocumentLine, SectionContent, JudgmentArtifact
流程編排：PipelineOrchestrator
  Phase 1: splitter_refactor.py  (讀取JSON, 解析段落, 產出 JudgmentArtifact)
  Phase 2: UltraStrict           (嚴格格式標記)
  Phase 3: Hybrid                (軟規則 + BERT)
  Phase 4: Adaptive              (學習規則 + 建立層級樹)
"""

import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==========================================
# Data Models (Context for the Pipeline)
# ==========================================

@dataclass(slots=True)
class SymbolDetection:
    """Stores information about a detected hierarchy symbol on a line"""
    detected_symbol: Optional[str] = None
    symbol_category: Optional[str] = None
    assigned_level: int = -1
    rule_based_score: float = 0.0
    bert_score: float = 0.0
    is_pua: bool = False
    method_used: str = ""
    is_learned_rule: bool = False

@dataclass(slots=True)
class DocumentLine:
    index: int
    original_text: str
    normalized_text: str
    tags: List[str] = field(default_factory=list)
    detection: Optional[SymbolDetection] = None
    
@dataclass(slots=True)
class SectionContent:
    name: str
    lines: List[DocumentLine] = field(default_factory=list)
    
@dataclass(slots=True)
class JudgmentArtifact:
    file_path: str
    full_lines: List[DocumentLine]
    sections: Dict[str, SectionContent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    key_lines: List[DocumentLine] = field(default_factory=list)
    hierarchy_tree: List[Dict[str, Any]] = field(default_factory=list)
    learned_rules: List[Dict[str, Any]] = field(default_factory=list)
    processing_stats: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# Pipeline Orchestrator
# ==========================================

class PipelineOrchestrator:
    def __init__(self, model_path: Optional[str] = None):
        print("🔧 初始化管線...")
        
        # Late imports to avoid circular dependencies
        from src.lchunk.detectors.ultra_strict import UltraStrictDetector
        from src.lchunk.detectors.soft_with_bert import HybridLevelSymbolDetector
        from src.lchunk.detectors.adaptive_hierarchy import AdaptiveHybridDetector

        self.ultra_strict_detector = UltraStrictDetector()
        self.hybrid_detector = HybridLevelSymbolDetector(model_path)
        self.adaptive_detector = AdaptiveHybridDetector(hybrid_detector=self.hybrid_detector)
        
        if self.hybrid_detector.is_model_loaded():
            print("✅ BERT 模型已載入，啟用完整的管線檢測")
        else:
            print("⚠️ 未載入 BERT 模型，只使用規則檢測管線")

    def process_file(self, file_path: Path) -> JudgmentArtifact:
        """處理單一檔案：貫穿整個 Pipeline"""
        # Late import to avoid circular dependency
        from src.lchunk.analyzers.splitter_refactor import split_judgment_document
        
        start_time = time.time()
        
        # Phase 1: Splitter — 讀取JSON、解析段落、產出 JudgmentArtifact
        artifact = split_judgment_document(file_path)
        if not artifact:
            raise ValueError(f"無法解析文件 {file_path} (可能缺少 JFULL 欄位)")
        
        # Phase 2: Ultra Strict Detection (In-place tagging)
        self.ultra_strict_detector.process_artifact(artifact)
        
        # Phase 3: Soft Rule + BERT Integration
        self.hybrid_detector.process_artifact(artifact)
        
        # Phase 4: Adaptive — Hierarchy Rule Learning and Tree Building
        self.adaptive_detector.process_artifact(artifact)
        
        processing_time = time.time() - start_time
        artifact.processing_stats["total_processing_time"] = processing_time
        
        return artifact

    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict:
        """批次處理整個目錄"""
        print(f"🔍 掃描目錄: {input_dir}")
        json_files = list(input_dir.glob("*.json"))
        print(f"📁 找到 {len(json_files)} 個 JSON 檔案")
        
        if not json_files:
            return {"error": "沒有找到 JSON 檔案"}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_stats = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_files": len(json_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0,
            "files": []
        }
        
        batch_start_time = time.time()
        
        for i, file_path in enumerate(json_files, 1):
            print(f"\n📄 處理檔案 {i}/{len(json_files)}: {file_path.name}")
            try:
                artifact = self.process_file(file_path)
                
                # Output result
                output_file = output_dir / f"{file_path.stem}_pipeline_result.json"
                
                # Prepare light-weight result for JSON
                result_data = {
                    "file_path": artifact.file_path,
                    "metadata": artifact.metadata,
                    "processing_stats": artifact.processing_stats,
                    "learned_rules": artifact.learned_rules,
                    "hierarchy_tree": artifact.hierarchy_tree,
                    "markers": [
                         {
                            "line": line.index,
                            "text": line.original_text,
                            "symbol": line.detection.detected_symbol,
                            "category": line.detection.symbol_category,
                            "level": line.detection.assigned_level,
                            "method": line.detection.method_used,
                            "bert_score": line.detection.bert_score
                         }
                         for line in artifact.full_lines if line.detection and line.detection.method_used not in ("empty_line", "rule_rejected")
                    ]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 成功 - 解析並儲存至 {output_file.name}")
                batch_stats["successful_files"] += 1
                
                batch_stats["files"].append({
                    "file_name": file_path.name,
                    "success": True,
                    "processing_time": artifact.processing_stats.get("total_processing_time", 0)
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ 失敗 - {e}")
                batch_stats["failed_files"] += 1
                batch_stats["files"].append({
                    "file_name": file_path.name,
                    "success": False,
                    "error": str(e)
                })
        
        batch_stats["total_processing_time"] = time.time() - batch_start_time
        
        # Save batch stats
        stats_file = output_dir / "pipeline_batch_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
            
        print(f"\n🎉 批次處理完成！結果儲存於 {output_dir}")
        return batch_stats

def main():
    parser = argparse.ArgumentParser(description="文件階層分析管線 (Splitter -> UltraStrict -> Hybrid -> Adaptive)")
    parser.add_argument("input_dir", help="輸入目錄路徑（包含 .json 判決文件）")
    parser.add_argument("--output", "-o", default="output/pipeline_results", help="輸出目錄")
    parser.add_argument("--model", "-m", help="BERT 模型路徑", default="models/bert/level_detector/best_model")
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ 錯誤：輸入目錄不存在或不是目錄: {input_path}")
        return
        
    orchestrator = PipelineOrchestrator(model_path=args.model)
    orchestrator.process_directory(input_path, output_path)

if __name__ == "__main__":
    main()
