#!/usr/bin/env python3
"""
自適應混合層級符號檢測器 (Adaptive Hybrid Detector)
"先學習再應用" 原則：文件分塊 → 規則學習 → 全文應用

Pipeline-integrated 版本:
  接收 JudgmentArtifact (已經由 Pipeline Splitter 解析完畢，且由 UltraStrict + Hybrid 標記過)
  1. 從 artifact.sections 中找出學習區間 (R-D / S-D)
  2. 在學習區間建立層級規則 (symbol_category -> level)
  3. 將規則套用至全文，為每個正類 DocumentLine.detection 設定 assigned_level
  4. 建立 line-based chunks 與 hierarchy tree，回寫至 artifact
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# 管線資料結構
from src.lchunk.pipeline import JudgmentArtifact, SymbolDetection, DocumentLine, SectionContent

# 導入現有模組
from .soft_with_bert import HybridLevelSymbolDetector, HybridDetectionResult
from src.lchunk.analyzers.splitter_refactor import normalize_text

logger = logging.getLogger(__name__)

# ==========================================
# Data Models
# ==========================================

@dataclass
class LevelingRule:
    """層級規則定義"""
    symbol_category: str
    assigned_level: int
    confidence: float
    learning_source: str  # "R-D", "S-D", "全文"
    occurrences: int
    examples: List[str]

@dataclass
class LineBasedChunk:
    """基於行的分塊結果"""
    level: int
    start_line: int
    end_line: int
    chunk_type: str  # "header", "date1", "sig", "date2", "appendix", "main_text", "facts", "reasons", "facts_and_reasons", "content", "leveling_symbol"
    content_lines: List[str]
    leveling_symbol: Optional[str] = None
    chunk_id: str = ""

@dataclass
class AdaptiveDetectionResult:
    """自適應檢測結果 (舊版介面用)"""
    filename: str
    file_structure: Dict
    learning_region: str
    learned_rules: List[LevelingRule]
    full_detection_results: List[HybridDetectionResult]
    applied_hierarchy: Dict
    processing_stats: Dict
    line_based_chunks: Optional[List[LineBasedChunk]] = None


# ==========================================
# Adaptive Hybrid Detector
# ==========================================

class AdaptiveHybridDetector:
    """自適應混合層級符號檢測器"""
    
    def __init__(self, model_path: Optional[str] = None, hybrid_detector: Optional[HybridLevelSymbolDetector] = None):
        if hybrid_detector is not None:
            self.hybrid_detector = hybrid_detector
        else:
            self.hybrid_detector = HybridLevelSymbolDetector(model_path)
        
        self.detection_results = []
        
        logger.debug("Adaptive hybrid detector initialized (chunking -> rule learning -> application).")

    # ==========================================
    # Pipeline-integrated entry point
    # ==========================================

    def process_artifact(self, artifact: JudgmentArtifact) -> None:
        """Pipeline 入口：處理已被 UltraStrict + Hybrid 標記過的 JudgmentArtifact。
        
        流程：
        1. 根據 artifact.sections 決定學習區間 (R-D / S-D / 全文)
        2. 在學習區間學習層級規則
        3. 將規則套用至全文：設定每個正類行的 assigned_level
        4. 建立 LineBasedChunk + Hierarchy Tree
        5. 回寫至 artifact.hierarchy_tree, artifact.learned_rules, artifact.processing_stats
        """
        # 步驟 1：決定學習區間
        learning_region, learning_doc_lines = self._determine_learning_region(artifact)
        artifact.processing_stats['learning_region'] = learning_region
        artifact.processing_stats['learning_lines_count'] = len(learning_doc_lines)
        
        # 步驟 2：在學習區間學習規則 — 按出現順序建立 symbol_category -> level 映射
        learned_rules = self._learn_rules_from_artifact(learning_doc_lines, learning_region)
        artifact.learned_rules = [
            {
                'symbol_category': rule.symbol_category,
                'assigned_level': rule.assigned_level,
                'confidence': rule.confidence,
                'learning_source': rule.learning_source,
                'occurrences': rule.occurrences,
                'examples': rule.examples,
            }
            for rule in learned_rules
        ]
        artifact.processing_stats['learned_rules_count'] = len(learned_rules)
        
        # 步驟 3：套用規則到全文
        rule_mapping = {rule.symbol_category: rule.assigned_level for rule in learned_rules}
        positive_count = 0
        unknown_categories = set()
        
        for doc_line in artifact.full_lines:
            detection = doc_line.detection
            if detection is None:
                continue
            
            # 只處理實際檢測到的正類
            if detection.method_used in ("ultra_strict_pua", "soft_rule_bert", "soft_rule_only"):
                if detection.detected_symbol is not None:
                    category = detection.symbol_category
                    if category in rule_mapping:
                        detection.assigned_level = rule_mapping[category]
                        detection.is_learned_rule = True
                    else:
                        detection.assigned_level = -1
                        detection.is_learned_rule = False
                        unknown_categories.add(category)
                    positive_count += 1
        
        artifact.processing_stats['rule_applied_positive'] = positive_count
        artifact.processing_stats['unknown_categories'] = list(unknown_categories)
        
        # 步驟 4：建立 LineBasedChunk
        chunks = self._create_line_based_chunks_from_artifact(artifact, learned_rules)
        
        # 步驟 5：建立 hierarchy tree
        tree = self._build_machine_tree(chunks)
        artifact.hierarchy_tree = tree
        artifact.processing_stats['line_based_chunks_count'] = len(chunks)
        artifact.processing_stats['hierarchy_tree_nodes'] = len(tree)

    # ==========================================
    # Internal methods for Pipeline mode
    # ==========================================

    def _determine_learning_region(self, artifact: JudgmentArtifact) -> Tuple[str, List[DocumentLine]]:
        """根據 artifact.sections 決定學習區間。
        
        優先順序：
        1. 有 facts_and_reasons → S-D：使用 facts_and_reasons 段落的行
        2. 有 reasons → R-D：使用 reasons 段落的行 
        3. 都沒有 → 全文
        """
        sections = artifact.sections
        
        # 先檢查 facts_and_reasons（S-D）
        far_section = sections.get("facts_and_reasons")
        if far_section and far_section.lines:
            return "S-D", far_section.lines
        
        # 再檢查 reasons（R-D）
        reasons_section = sections.get("reasons")
        if reasons_section and reasons_section.lines:
            return "R-D", reasons_section.lines
        
        # 都沒有就用全文
        return "全文", artifact.full_lines

    def _learn_rules_from_artifact(self, doc_lines: List[DocumentLine], learning_region: str) -> List[LevelingRule]:
        """在學習區間中，根據已標記的 detection 結果，建立 symbol_category → level 映射。
        
        完全動態學習：按照符號類型在學習區間中首次出現的順序，分配遞增層級。
        """
        category_levels: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        category_examples: Dict[str, List[str]] = {}
        current_level = 1
        
        for doc_line in doc_lines:
            detection = doc_line.detection
            if detection is None:
                continue
            
            # 只學習正類（含 ultra_strict 和 soft_rule 結果）
            if detection.detected_symbol is None:
                continue
            if detection.method_used in ("empty_line", "rule_rejected"):
                continue
            
            category = detection.symbol_category
            if category is None:
                continue
            
            if category not in category_levels:
                category_levels[category] = current_level
                category_counts[category] = 0
                category_examples[category] = []
                current_level += 1
            
            category_counts[category] += 1
            if len(category_examples[category]) < 3:
                text_preview = doc_line.original_text.strip()[:50] + '...'
                category_examples[category].append(text_preview)
        
        total_positive = sum(category_counts.values())
        
        rules = []
        for category, level in category_levels.items():
            count = category_counts[category]
            rule = LevelingRule(
                symbol_category=category,
                assigned_level=level,
                confidence=count / total_positive if total_positive > 0 else 0,
                learning_source=learning_region,
                occurrences=count,
                examples=category_examples.get(category, [])
            )
            rules.append(rule)
            logger.debug("  %s -> level %d (confidence %.3f)", category, level, rule.confidence)
        
        logger.info("Learned %d hierarchy rules within [%s] region.", len(rules), learning_region)
        return rules

    def _create_line_based_chunks_from_artifact(self, artifact: JudgmentArtifact, learned_rules: List[LevelingRule]) -> List[LineBasedChunk]:
        """基於 artifact.sections + 已標記的行 建立 line-based chunks。
        
        利用 artifact.sections 中已有的段落結構 (header, date1, sig, date2, appendix 等)
        以及各行的 detection 結果，建立分塊。

        appendix section 會按每個「附[錄件圖表]」標頭行拆成多個獨立 chunk（chunk_id=appendix_0, appendix_1,…）
        """
        chunks: List[LineBasedChunk] = []
        sections = artifact.sections

        # 規則映射
        rule_mapping = {rule.symbol_category: rule.assigned_level for rule in learned_rules}

        # 屬於結構標記（不參與內容層級分析）的 section 名稱
        special_section_names = {'header', 'date1', 'sig', 'date2', 'appendix'}
        structural_marker_names = {'main_text', 'facts', 'reasons', 'facts_and_reasons'}

        # 收集所有行的分類（正類層級符號行）
        leveling_lines: Dict[int, Tuple[str, str, int]] = {}  # index -> (symbol, category, level)
        for doc_line in artifact.full_lines:
            det = doc_line.detection
            if det and det.detected_symbol is not None and det.method_used not in ("empty_line", "rule_rejected"):
                category = det.symbol_category
                level = det.assigned_level if det.assigned_level > 0 else rule_mapping.get(category, 1)
                leveling_lines[doc_line.index] = (det.detected_symbol, category, level)

        # 收集特殊段落行號集合
        special_line_set: set = set()

        # ── header → level -3 ──────────────────────────────────────
        section = sections.get('header')
        if section and section.lines:
            start = section.lines[0].index
            end = section.lines[-1].index
            chunks.append(LineBasedChunk(
                level=-3, start_line=start, end_line=end,
                chunk_type='header',
                content_lines=[dl.original_text for dl in section.lines],
                chunk_id='header'
            ))
            for dl in section.lines:
                special_line_set.add(dl.index)

        # ── date1 → level -2（單行判決日期）───────────────────────
        section = sections.get('date1')
        if section and section.lines:
            start = section.lines[0].index
            end = section.lines[-1].index
            chunks.append(LineBasedChunk(
                level=-2, start_line=start, end_line=end,
                chunk_type='date1',
                content_lines=[dl.original_text for dl in section.lines],
                chunk_id='date1'
            ))
            for dl in section.lines:
                special_line_set.add(dl.index)

        # ── sig → level -2（單栏保留逐行）───────────────────────
        section = sections.get('sig')
        if section and section.lines:
            start = section.lines[0].index
            end = section.lines[-1].index
            chunks.append(LineBasedChunk(
                level=-2, start_line=start, end_line=end,
                chunk_type='sig',
                content_lines=[dl.original_text for dl in section.lines],
                chunk_id='sig'
            ))
            for dl in section.lines:
                special_line_set.add(dl.index)

        # ── date2 → level -2（正本送達日期）──────────────────────
        section = sections.get('date2')
        if section and section.lines:
            start = section.lines[0].index
            end = section.lines[-1].index
            chunks.append(LineBasedChunk(
                level=-2, start_line=start, end_line=end,
                chunk_type='date2',
                content_lines=[dl.original_text for dl in section.lines],
                chunk_id='date2'
            ))
            for dl in section.lines:
                special_line_set.add(dl.index)

        # ── appendix → 按附錄標頭切分為多個獨立 level 0 chunk ───
        # 每遇「附[錄件圖表]」開頭的行，開始一個新附錄 chunk
        _appendix_pat = re.compile(r'^附[錄件圖表]')
        section = sections.get('appendix')
        if section and section.lines:
            current_appendix_lines: List[DocumentLine] = []
            appendix_chunk_index = 0

            def _flush_appendix():
                nonlocal current_appendix_lines, appendix_chunk_index
                if current_appendix_lines:
                    s = current_appendix_lines[0].index
                    e = current_appendix_lines[-1].index
                    chunks.append(LineBasedChunk(
                        level=0, start_line=s, end_line=e,
                        chunk_type='appendix',
                        content_lines=[dl.original_text for dl in current_appendix_lines],
                        chunk_id=f'appendix_{appendix_chunk_index}'
                    ))
                    appendix_chunk_index += 1
                    current_appendix_lines = []

            for dl in section.lines:
                cleaned = dl.normalized_text or normalize_text(dl.original_text)
                if _appendix_pat.match(cleaned) and current_appendix_lines:
                    # 遇到新附錄標頭，flush 当前 chunk
                    _flush_appendix()
                current_appendix_lines.append(dl)
                special_line_set.add(dl.index)
            _flush_appendix()  # 收尾

        
        # 處理結構標記行 (主文、事實、理由 等) — 層級 0
        for section_name in structural_marker_names:
            # 尋找對應的 key_line
            for key_line in artifact.key_lines:
                normalized = key_line.normalized_text
                if section_name == 'main_text' and normalized == '主文':
                    chunks.append(LineBasedChunk(
                        level=0, start_line=key_line.index, end_line=key_line.index,
                        chunk_type="main_text", content_lines=[key_line.original_text],
                        chunk_id=f"main_text_{key_line.index}"
                    ))
                    special_line_set.add(key_line.index)
                elif section_name == 'facts' and normalized == '事實':
                    chunks.append(LineBasedChunk(
                        level=0, start_line=key_line.index, end_line=key_line.index,
                        chunk_type="facts", content_lines=[key_line.original_text],
                        chunk_id=f"facts_{key_line.index}"
                    ))
                    special_line_set.add(key_line.index)
                elif section_name == 'reasons' and normalized == '理由':
                    chunks.append(LineBasedChunk(
                        level=0, start_line=key_line.index, end_line=key_line.index,
                        chunk_type="reasons", content_lines=[key_line.original_text],
                        chunk_id=f"reasons_{key_line.index}"
                    ))
                    special_line_set.add(key_line.index)
                elif section_name == 'facts_and_reasons' and '事實' in normalized and '理由' in normalized:
                    chunks.append(LineBasedChunk(
                        level=0, start_line=key_line.index, end_line=key_line.index,
                        chunk_type="facts_and_reasons", content_lines=[key_line.original_text],
                        chunk_id=f"facts_and_reasons_{key_line.index}"
                    ))
                    special_line_set.add(key_line.index)
        
        # 處理內容區域：在非特殊段落中，收集層級符號行和它們之間的內容行
        content_sections = ['main_text', 'facts', 'reasons', 'facts_and_reasons']
        content_lines_range = []
        for sname in content_sections:
            section = sections.get(sname)
            if section and section.lines:
                for dl in section.lines:
                    if dl.index not in special_line_set:
                        content_lines_range.append(dl.index)
        
        content_lines_range.sort()
        
        # 拿出層級符號行的 chunk 和中間的內容 chunk
        current_content_indices: List[int] = []
        
        def flush_content():
            nonlocal current_content_indices
            if current_content_indices:
                c_lines = [artifact.full_lines[idx].original_text for idx in current_content_indices]
                chunks.append(LineBasedChunk(
                    level=-1,
                    start_line=current_content_indices[0],
                    end_line=current_content_indices[-1],
                    chunk_type="content",
                    content_lines=c_lines,
                    chunk_id=f"content_{current_content_indices[0]}_{current_content_indices[-1]}"
                ))
                current_content_indices = []
        
        for idx in content_lines_range:
            if idx in leveling_lines:
                # 先 flush 之前的內容
                flush_content()
                
                symbol, category, level = leveling_lines[idx]
                chunks.append(LineBasedChunk(
                    level=level,
                    start_line=idx,
                    end_line=idx,
                    chunk_type="leveling_symbol",
                    content_lines=[artifact.full_lines[idx].original_text],
                    leveling_symbol=symbol,
                    chunk_id=f"level_{level}_{idx}"
                ))
            else:
                current_content_indices.append(idx)
        
        flush_content()
        
        # 按行號排序
        chunks.sort(key=lambda x: x.start_line)
        
        logger.info("Constructed %d line-based chunks from artifact.", len(chunks))
        return chunks

    @staticmethod
    def _chunk_to_machine_node(chunk: LineBasedChunk) -> Dict[str, Any]:
        """將分塊轉換為機器可讀節點。"""
        return {
            "level": chunk.level,
            "chunk_type": chunk.chunk_type,
            "start_line": chunk.start_line + 1,
            "end_line": chunk.end_line + 1,
            "content_lines": list(chunk.content_lines),
            "leveling_symbol": chunk.leveling_symbol,
            "chunk_id": chunk.chunk_id,
            "children": [],
        }

    def _build_machine_tree(self, chunks: List[LineBasedChunk]) -> List[Dict[str, Any]]:
        """建立機器可讀層級樹，並將 Lv -1 內容附加到對應的上層 (Lv >= 1)。"""
        if not chunks:
            return []

        ordered_chunks = sorted(chunks, key=lambda item: item.start_line)
        tree: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []

        for chunk in ordered_chunks:
            if chunk.level >= 0:
                node = self._chunk_to_machine_node(chunk)
                while stack and stack[-1]["level"] >= chunk.level:
                    stack.pop()

                if stack:
                    stack[-1]["children"].append(node)
                else:
                    tree.append(node)

                stack.append(node)
            elif chunk.level == -1:
                recipient: Optional[Dict[str, Any]] = None
                for candidate in reversed(stack):
                    if candidate["level"] >= 1:
                        recipient = candidate
                        break

                if recipient is not None:
                    recipient["content_lines"].extend(chunk.content_lines)
                    recipient["end_line"] = max(recipient["end_line"], chunk.end_line + 1)
                else:
                    tree.append(self._chunk_to_machine_node(chunk))
            else:
                # Header/Footer/Sig/Date 等保留為獨立節點
                tree.append(self._chunk_to_machine_node(chunk))

        return tree




def main():
    """主函數 - 自適應檢測演示"""
    logger.info("Adaptive detector demo following the 'learn then apply' principle.")
    
    model_path = "models/bert/level_detector/best_model"
    detector = AdaptiveHybridDetector(model_path if Path(model_path).exists() else None)

    # sample_dir = Path("data/processed/sample")
    # detector.process_sample_directory(sample_dir, verbose=False)

if __name__ == "__main__":
    main()
