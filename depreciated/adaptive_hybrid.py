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
    chunk_type: str  # "header", "main_text", "facts", "reasons", "facts_and_reasons", "footer", "content", "leveling_symbol"
    content_lines: List[str]
    leveling_symbol: Optional[str] = None
    chunk_id: str = ""



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
        
        利用 artifact.sections 中已有的段落結構 (header, main_text, sig, footer, appendix 等) 
        以及各行的 detection 結果，建立分塊。
        """
        chunks: List[LineBasedChunk] = []
        sections = artifact.sections
        
        # 規則映射
        rule_mapping = {rule.symbol_category: rule.assigned_level for rule in learned_rules}
        
        # 收集「特殊段落」的行索引 (header, sig, footer, appendix 屬於結構標記)
        special_section_names = {'header', 'sig', 'footer', 'appendix'}
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
        
        # 處理特殊段落
        for section_name in special_section_names:
            section = sections.get(section_name)
            if section and section.lines:
                start = section.lines[0].index
                end = section.lines[-1].index
                
                # header 和 footer 層級 -3
                if section_name in ('header', 'footer'):
                    chunks.append(LineBasedChunk(
                        level=-3,
                        start_line=start,
                        end_line=end,
                        chunk_type=section_name,
                        content_lines=[dl.original_text for dl in section.lines],
                        chunk_id=section_name
                    ))
                    for dl in section.lines:
                        special_line_set.add(dl.index)
                
                # sig 層級 -2
                elif section_name == 'sig':
                    chunks.append(LineBasedChunk(
                        level=-2,
                        start_line=start,
                        end_line=end,
                        chunk_type="sig",
                        content_lines=[dl.original_text for dl in section.lines],
                        chunk_id="sig"
                    ))
                    for dl in section.lines:
                        special_line_set.add(dl.index)
                
                # appendix 層級 0
                elif section_name == 'appendix':
                    chunks.append(LineBasedChunk(
                        level=0,
                        start_line=start,
                        end_line=end,
                        chunk_type="appendix",
                        content_lines=[dl.original_text for dl in section.lines],
                        chunk_id="appendix"
                    ))
                    for dl in section.lines:
                        special_line_set.add(dl.index)
        
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

    # ==========================================
    # Legacy methods (kept for backward compatibility)
    # ==========================================
    
    def detect_special_markers(self, lines: List[str]) -> Dict[str, List[int]]:
        """檢測特殊標記 — 舊版介面"""
        from src.lchunk.analyzers.splitter_refactor import find_section_pattern
        
        markers = {
            'main_text': [],
            'reasons': [],
            'facts': [],
            'facts_and_reasons': [],
            'appendix': [],
            'dates': []
        }
        
        patterns = find_section_pattern()
        
        date_lines = []
        for line_num, line in enumerate(lines):
            line_text = line.strip()
            if not line_text:
                continue
            if patterns['date_pattern'].search(line_text) or patterns['date_pattern_strict'].search(line_text):
                date_lines.append(line_num)
                markers['dates'].append(line_num)
        
        last_date_line = max(date_lines) if date_lines else None
        
        for line_num, line in enumerate(lines):
            line_text = line.strip()
            if not line_text:
                continue
            
            normalized_text = line_text.replace(' ', '').replace('\u3000', '').replace('\t', '')
            
            if normalized_text == '主文':
                markers['main_text'].append(line_num)
            elif normalized_text == '事實':
                markers['facts'].append(line_num)
            elif normalized_text == '理由':
                markers['reasons'].append(line_num)
            elif normalized_text in ['事實及理由', '事實和理由']:
                markers['facts_and_reasons'].append(line_num)
            elif last_date_line is not None and line_num > last_date_line:
                if len(normalized_text) >= 2:
                    first_two_chars = normalized_text[:2]
                    if first_two_chars in ['附錄', '附件', '附圖', '附表']:
                        markers['appendix'].append(line_num)
        
        return markers
    
    def create_line_based_chunks(self, lines: List[str], detection_results: List[HybridDetectionResult], 
                                learned_rules: List[LevelingRule]) -> List[LineBasedChunk]:
        """基於行的分塊方法 — 舊版介面"""
        level_mapping = {}
        for rule in learned_rules:
            level_mapping[rule.symbol_category] = rule.assigned_level

        special_markers = self.detect_special_markers(lines)
        
        special_line_set = set()
        for marker_lines in special_markers.values():
            special_line_set.update(marker_lines)
        
        chunks: List[LineBasedChunk] = []

        def emit_content_segment(segment_indices: List[int]):
            if not segment_indices:
                return
            segment_lines = [lines[idx] for idx in segment_indices]
            chunks.append(LineBasedChunk(
                level=-1,
                start_line=segment_indices[0],
                end_line=segment_indices[-1],
                chunk_type="content",
                content_lines=segment_lines,
                chunk_id=f"content_{segment_indices[0]}_{segment_indices[-1]}"
            ))

        leveling_symbol_lines = {}
        for result in detection_results:
            if result.final_prediction:
                symbol_category = result.symbol_category
                assigned_level = level_mapping.get(symbol_category, 1)
                leveling_symbol_lines[result.line_number - 1] = (
                    result.detected_symbol, symbol_category, assigned_level
                )
        
        main_text_line = special_markers['main_text'][0] if special_markers['main_text'] else None
        last_date_line = max(special_markers['dates']) if special_markers['dates'] else None
        
        if main_text_line is not None and main_text_line > 0:
            header_content = lines[:main_text_line]
            chunks.append(LineBasedChunk(
                level=-3, start_line=0, end_line=main_text_line - 1,
                chunk_type="header", content_lines=header_content, chunk_id="header"
            ))
        
        content_end = len(lines) - 1
        first_appendix_line = min(special_markers['appendix']) if special_markers['appendix'] else None
        if first_appendix_line is not None:
            content_end = first_appendix_line - 1
        
        content_start = main_text_line if main_text_line is not None else 0
        
        for marker_type, line_numbers in special_markers.items():
            for line_num in line_numbers:
                if content_start <= line_num <= content_end:
                    if marker_type == 'dates':
                        chunk_level = -2
                        chunk_type = 'date'
                    else:
                        chunk_level = 0
                        chunk_type = marker_type
                    
                    chunks.append(LineBasedChunk(
                        level=chunk_level, start_line=line_num, end_line=line_num,
                        chunk_type=chunk_type, content_lines=[lines[line_num]],
                        chunk_id=f"{chunk_type}_{line_num}"
                    ))
        
        sorted_symbol_lines = sorted(leveling_symbol_lines.keys())
        
        def collect_content_segments(start_idx: int, end_idx: int):
            current_indices: List[int] = []
            for idx in range(start_idx, end_idx):
                if idx in special_line_set or idx in leveling_symbol_lines:
                    if current_indices:
                        emit_content_segment(current_indices)
                        current_indices = []
                    continue
                current_indices.append(idx)
            if current_indices:
                emit_content_segment(current_indices)
        
        current_pos = content_start
        for symbol_line in sorted_symbol_lines:
            if symbol_line < content_start or symbol_line > content_end:
                continue
            
            if current_pos < symbol_line:
                collect_content_segments(current_pos, symbol_line)
            
            symbol, category, level = leveling_symbol_lines[symbol_line]
            chunks.append(LineBasedChunk(
                level=level, start_line=symbol_line, end_line=symbol_line,
                chunk_type="leveling_symbol", content_lines=[lines[symbol_line]],
                leveling_symbol=symbol, chunk_id=f"level_{level}_{symbol_line}"
            ))
            
            current_pos = symbol_line + 1
        
        if current_pos <= content_end:
            collect_content_segments(current_pos, content_end + 1)
        
        if last_date_line is not None:
            footer_start = last_date_line + 1
            footer_end = first_appendix_line - 1 if first_appendix_line else len(lines) - 1
            
            if footer_start <= footer_end:
                footer_lines = []
                for line_idx in range(footer_start, footer_end + 1):
                    if line_idx not in special_line_set and line_idx not in leveling_symbol_lines:
                        footer_lines.append(lines[line_idx])
                
                if footer_lines:
                    chunks.append(LineBasedChunk(
                        level=-3, start_line=footer_start, end_line=footer_end,
                        chunk_type="footer", content_lines=footer_lines, chunk_id="footer"
                    ))
        
        chunks.sort(key=lambda x: x.start_line)
        return chunks

    def build_machine_tree(self, chunks: List[LineBasedChunk]) -> List[Dict[str, Any]]:
        """建立機器可讀層級樹 — 舊版介面（委派到內部方法）"""
        return self._build_machine_tree(chunks)

    def build_machine_payload(self, result: AdaptiveDetectionResult) -> Dict[str, Any]:
        """組裝機器可讀輸出所需的 payload — 舊版介面"""
        machine_tree = self.build_machine_tree(result.line_based_chunks or [])
        learned_rules = [
            {
                "symbol_category": rule.symbol_category,
                "assigned_level": rule.assigned_level,
                "confidence": rule.confidence,
                "learning_source": rule.learning_source,
                "occurrences": rule.occurrences,
                "examples": rule.examples,
            }
            for rule in result.learned_rules
        ]

        return {
            "filename": result.filename,
            "file_structure": result.file_structure,
            "learning_region": result.learning_region,
            "processing_stats": result.processing_stats,
            "learned_rules": learned_rules,
            "applied_hierarchy": result.applied_hierarchy,
            "hierarchy": machine_tree,
            "line_based_chunks": [
                {
                    "level": chunk.level,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "content_lines": list(chunk.content_lines),
                    "leveling_symbol": chunk.leveling_symbol,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in result.line_based_chunks or []
            ],
        }

    def export_machine_result(self, result: AdaptiveDetectionResult, output_dir: Path) -> Path:
        """輸出機器可讀 JSON 檔 — 舊版介面"""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.build_machine_payload(result)

        target_name = f"{Path(result.filename).stem}_machine.json"
        target_path = output_dir / target_name
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target_path
    
    def concatenate_level_content(self, chunks: List[LineBasedChunk]) -> Dict[str, List[str]]:
        """合併相同層級的內容 — 舊版介面"""
        level_content = {}
        
        for chunk in chunks:
            level_key = f"Lv_{chunk.level}"
            if level_key not in level_content:
                level_content[level_key] = []
            
            chunk_info = f"[{chunk.chunk_type}:{chunk.start_line+1}-{chunk.end_line+1}]"
            level_content[level_key].append(chunk_info)
            level_content[level_key].extend(chunk.content_lines)
        
        return level_content
    
    def analyze_file_structure(self, file_path: Path) -> Tuple[bool, Dict]:
        """分析檔案結構 — 舊版介面"""
        try:
            from src.lchunk.analyzers.splitter_refactor import process_single_file, find_section_pattern
            
            success, result = process_single_file(file_path)
            
            if not success:
                return False, {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sections = result.get('sections', {}) if isinstance(result, dict) else {}
            has_main_text = bool(sections.get('main_text', []))
            has_facts = bool(sections.get('facts', []))
            has_reasons = bool(sections.get('reasons', []))
            has_facts_and_reasons = bool(sections.get('facts_and_reasons', []))
            
            learning_region = None
            learning_lines = []
            
            if has_facts_and_reasons:
                learning_region = "S-D"
                fr_lines = sections.get('facts_and_reasons', [])
                if fr_lines:
                    full_lines = data['JFULL'].split('\n')
                    fr_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [l.strip() for l in fr_lines[:3]]:
                            fr_start_line = i
                            break
                    if fr_start_line is not None:
                        learning_lines = full_lines[fr_start_line:]
            
            elif has_reasons:
                learning_region = "R-D"
                reasons_lines = sections.get('reasons', [])
                if reasons_lines:
                    full_lines = data['JFULL'].split('\n')
                    reasons_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [l.strip() for l in reasons_lines[:3]]:
                            reasons_start_line = i
                            break
                    if reasons_start_line is not None:
                        learning_lines = full_lines[reasons_start_line:]
            
            if not learning_region:
                learning_region = "全文"
                learning_lines = data['JFULL'].split('\n')
            
            structure_info = {
                'sections': sections,
                'has_main_text': has_main_text,
                'has_facts': has_facts,
                'has_reasons': has_reasons,
                'has_facts_and_reasons': has_facts_and_reasons,
                'learning_region': learning_region,
                'learning_lines': learning_lines,
                'full_text_lines': data['JFULL'].split('\n'),
                'total_lines': len(data['JFULL'].split('\n'))
            }
            
            return True, structure_info
        
        except Exception as exc:
            logger.exception("Failed to analyze file structure for %s", file_path)
            return False, {}
    
    def learn_leveling_rules(self, learning_lines: List[str], learning_region: str, verbose: bool = False) -> List[LevelingRule]:
        """在學習區間建立層級規則 — 舊版介面"""
        learning_results = self.hybrid_detector.detect_hybrid_markers(learning_lines, verbose=verbose)
        
        self.hybrid_detector.detection_results = learning_results
        hierarchy_analysis = self.hybrid_detector.detect_hierarchy_levels()
        
        if not hierarchy_analysis or not hierarchy_analysis.get('level_mapping'):
            return []
        
        rules = []
        level_mapping = hierarchy_analysis['level_mapping']
        
        for symbol_category, level_info in level_mapping.items():
            rule = LevelingRule(
                symbol_category=symbol_category,
                assigned_level=level_info['assigned_level'],
                confidence=level_info['count'] / len([r for r in learning_results if r.final_prediction]),
                learning_source=learning_region,
                occurrences=level_info['count'],
                examples=[ex['text'][:50] + '...' for ex in level_info['examples'][:3]]
            )
            rules.append(rule)
        
        return rules

    def apply_leveling_rules(self, full_results: List[HybridDetectionResult], 
                           learned_rules: List[LevelingRule]) -> Dict:
        """將學習到的規則應用到全文檢測結果 — 舊版介面"""
        rule_mapping = {}
        for rule in learned_rules:
            rule_mapping[rule.symbol_category] = rule.assigned_level
        
        enhanced_hierarchy = []
        unknown_categories = set()
        
        for result in full_results:
            if not result.final_prediction:
                continue
            
            symbol_category = result.symbol_category
            
            if symbol_category in rule_mapping:
                assigned_level = rule_mapping[symbol_category]
            else:
                unknown_categories.add(symbol_category)
                assigned_level = -1
            
            enhanced_hierarchy.append({
                'line_number': result.line_number,
                'detected_symbol': result.detected_symbol,
                'symbol_category': symbol_category,
                'assigned_level': assigned_level,
                'is_learned_rule': symbol_category not in unknown_categories,
                'line_text': result.line_text,
                'method_used': result.method_used,
                'bert_score': result.bert_score
            })
        
        level_stats = {}
        for item in enhanced_hierarchy:
            category = item['symbol_category']
            if category not in level_stats:
                level_stats[category] = {
                    'assigned_level': item['assigned_level'],
                    'count': 0,
                    'is_learned': item['is_learned_rule'],
                    'examples': []
                }
            level_stats[category]['count'] += 1
            if len(level_stats[category]['examples']) < 3:
                level_stats[category]['examples'].append({
                    'line': item['line_number'],
                    'symbol': item['detected_symbol'],
                    'text': item['line_text'][:50] + '...'
                })
        
        return {
            'enhanced_hierarchy': enhanced_hierarchy,
            'level_mapping': level_stats,
            'rule_coverage': (len(rule_mapping) - len(unknown_categories)) / len(rule_mapping) if rule_mapping else 0,
            'total_levels': len(set(item['assigned_level'] for item in enhanced_hierarchy)),
            'total_symbols': len(enhanced_hierarchy)
        }
    
    def process_single_file(self, file_path: Path, verbose: bool = False) -> Optional[AdaptiveDetectionResult]:
        """處理單個檔案 — 舊版介面（完整的自適應檢測流程 + 基於行的分塊）"""
        if verbose:
            logger.info("Processing file: %s", file_path.name)
        
        success, structure_info = self.analyze_file_structure(file_path)
        if not success:
            return None
        
        learning_region = structure_info['learning_region']
        
        full_text_lines = structure_info['full_text_lines']
        full_detection_results = self.hybrid_detector.detect_hybrid_markers(full_text_lines, verbose=verbose)
        
        learning_lines = structure_info['learning_lines']
        learned_rules = self.learn_leveling_rules(learning_lines, learning_region, verbose=verbose)
        
        applied_hierarchy = self.apply_leveling_rules(full_detection_results, learned_rules)
        
        line_based_chunks = self.create_line_based_chunks(full_text_lines, full_detection_results, learned_rules)
        
        level_content = self.concatenate_level_content(line_based_chunks)
        
        processing_stats = {
            'total_lines': structure_info['total_lines'],
            'learning_lines': len(learning_lines),
            'total_symbols_detected': len([r for r in full_detection_results if r.final_prediction]),
            'learned_rules_count': len(learned_rules),
            'rule_coverage': applied_hierarchy['rule_coverage'],
            'final_levels': applied_hierarchy['total_levels'],
            'line_based_chunks_count': len(line_based_chunks),
            'level_content_summary': {k: len([line for line in v if not line.startswith('[')]) 
                                    for k, v in level_content.items()}
        }
        
        result = AdaptiveDetectionResult(
            filename=file_path.name,
            file_structure=structure_info,
            learning_region=learning_region,
            learned_rules=learned_rules,
            full_detection_results=full_detection_results,
            applied_hierarchy=applied_hierarchy,
            processing_stats=processing_stats,
            line_based_chunks=line_based_chunks
        )
        
        return result
    
    def process_sample_directory(
        self,
        sample_dir: Path,
        output_dir: Optional[Path] = None,
        max_files: Optional[int] = None,
        verbose: bool = False,
        generate_reports: bool = True,
    ):
        """處理 sample 目錄 — 舊版介面"""
        json_files = sorted(p for p in sample_dir.glob("*.json") if p.is_file())
        if max_files is not None:
            json_files = json_files[:max_files]
        if not json_files:
            return
        
        all_results = []
        learning_region_stats = {'S-D': 0, 'R-D': 0, '全文': 0}
        exported_files = []
        
        print(f"Processing {len(json_files)} files...")
        
        for i, json_file in enumerate(json_files, 1):
            if not verbose:
                progress = f"[{i}/{len(json_files)}] {json_file.name}"
                print(f"\r{progress:<60}", end="", flush=True)
            
            result = self.process_single_file(json_file, verbose=verbose)
            if result:
                all_results.append(result)
                learning_region_stats[result.learning_region] += 1
                
                try:
                    export_path = self.export_machine_result(result, output_dir or Path("output"))
                    exported_files.append(export_path)
                except Exception as exc:
                    if verbose:
                        logger.error("Failed to export for %s: %s", json_file.name, exc)
        
        if not verbose:
            print()
        
        if generate_reports:
            self._generate_batch_report(all_results, learning_region_stats, output_dir)
        
        print(f"✅ Completed: {len(all_results)} files processed, {len(exported_files)} machine results exported")
    
    def _generate_batch_report(
        self,
        results: List[AdaptiveDetectionResult],
        region_stats: Dict,
        output_dir: Optional[Path] = None,
    ):
        """生成批量處理報告 — 舊版介面"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(output_dir) if output_dir else Path("output")
        target_dir.mkdir(parents=True, exist_ok=True)
        report_file = target_dir / f"adaptive_detection_report_{timestamp}.md"
        
        report = f"""# 自適應混合層級符號檢測報告
生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
處理檔案: {len(results)} 個

## 📊 整體統計

### 學習區間分布
- **S-D 區間** (事實理由合併): {region_stats['S-D']} 檔案
- **R-D 區間** (理由章節): {region_stats['R-D']} 檔案  
- **全文檢測**: {region_stats['全文']} 檔案

### 處理統計
"""
        
        if results:
            total_lines = sum(r.processing_stats['total_lines'] for r in results)
            total_symbols = sum(r.processing_stats['total_symbols_detected'] for r in results)
            avg_coverage = sum(r.processing_stats['rule_coverage'] for r in results) / len(results)
            
            report += f"- 總行數: {total_lines:,}\n"
            report += f"- 總符號數: {total_symbols:,}\n"
            report += f"- 平均規則覆蓋率: {avg_coverage:.1%}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Report saved to %s", report_file)


def main():
    """主函數 - 自適應檢測演示"""
    logger.info("Adaptive detector demo following the 'learn then apply' principle.")
    
    model_path = "models/bert/level_detector/best_model"
    detector = AdaptiveHybridDetector(model_path if Path(model_path).exists() else None)

    sample_dir = Path("data/processed/sample")
    detector.process_sample_directory(sample_dir, verbose=False)

if __name__ == "__main__":
    main()
