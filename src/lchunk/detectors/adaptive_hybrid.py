#!/usr/bin/env python3
"""
自適應混合層級符號檢測器 (Adaptive Hybrid Detector)
先學習再應用" 原則：文件分塊 → 規則學習 → 全文應用

處理流程：
1. 文件分塊：使用 comprehensive_analysis 分析文件結構
2. 全文層級符號偵測：用 hybrid_detector 檢測所有符號
3. 規則學習區間：在 R-D 或 S-D 區間建立層級規則
4. 層級規則建立：分析符號類型和層級模式
5. 全文應用：將學習到的規則應用到整個文件

"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# 導入現有模組
sys.path.append(".")
from .hybrid import HybridLevelSymbolDetector, HybridDetectionResult
from ..analyzers.splitter import process_single_file, find_section_patterns
from ..analyzers.comprehensive import analyze_filtered_dataset


logger = logging.getLogger(__name__)


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
    leveling_symbol: Optional[str] = None  # 如果是層級符號行
    chunk_id: str = ""


@dataclass
class AdaptiveDetectionResult:
    """自適應檢測結果"""

    filename: str
    file_structure: Dict  # comprehensive_analysis 結果
    learning_region: str  # "R-D", "S-D", "全文"
    learned_rules: List[LevelingRule]
    full_detection_results: List[HybridDetectionResult]
    applied_hierarchy: Dict
    processing_stats: Dict
    line_based_chunks: Optional[List[LineBasedChunk]] = None  # 新增：基於行的分塊結果


class AdaptiveHybridDetector:
    """自適應混合層級符號檢測器"""

    def __init__(self, model_path: Optional[str] = None):
        # 初始化基礎檢測器 - 只在有模型時才載入 BERT
        self.hybrid_detector = HybridLevelSymbolDetector(model_path)

        # 自適應檢測結果
        self.detection_results = []

        logger.debug(
            "Adaptive hybrid detector initialized (chunking -> rule learning -> application)."
        )

    def detect_special_markers(self, lines: List[str]) -> Dict[str, List[int]]:
        """檢測特殊標記：主文(lv 0), 理由(lv 0), 事實(lv 0), 事實及理由(lv 0), 附錄(lv 0), date1(lv -2), date2(lv -2)"""
        markers = {
            "main_text": [],  # 主文 (lv 0)
            "reasons": [],  # 理由 (lv 0)
            "facts": [],  # 事實 (lv 0)
            "facts_and_reasons": [],  # 事實及理由 (lv 0)
            "appendix": [],  # 附錄 (lv 0) - 新增
            "dates": [],  # 日期 (lv -2)
        }

        patterns = find_section_patterns()

        # 第一階段：檢測所有日期行以確定Date2位置
        date_lines = []
        for line_num, line in enumerate(lines):
            line_text = line.strip()
            if not line_text:
                continue
            if patterns["date_pattern"].search(line_text) or patterns[
                "date_pattern_strict"
            ].search(line_text):
                date_lines.append(line_num)
                markers["dates"].append(line_num)
                logger.debug(
                    "Detected 'date' marker at line %d: %s", line_num + 1, line_text
                )

        # 確定最後一個日期行（Date2）
        last_date_line = max(date_lines) if date_lines else None

        # 第二階段：檢測其他特殊標記
        for line_num, line in enumerate(lines):
            line_text = line.strip()
            if not line_text:
                continue

            # 標準化文字：移除所有空白字符（半形空白、全形空白、tab等）
            normalized_text = (
                line_text.replace(" ", "").replace("　", "").replace("\t", "")
            )

            # 檢測主文 (支援 "主文", "主　文", "主 文" 等格式)
            if normalized_text == "主文":
                markers["main_text"].append(line_num)
                logger.debug(
                    "Detected 'main_text' marker at line %d: %s",
                    line_num + 1,
                    line_text,
                )

            # 檢測事實 (支援 "事實", "事　實", "事 實" 等格式)
            elif normalized_text == "事實":
                markers["facts"].append(line_num)
                logger.debug(
                    "Detected 'facts' marker at line %d: %s", line_num + 1, line_text
                )

            # 檢測理由 (支援 "理由", "理　由", "理 由" 等格式)
            elif normalized_text == "理由":
                markers["reasons"].append(line_num)
                logger.debug(
                    "Detected 'reasons' marker at line %d: %s", line_num + 1, line_text
                )

            # 檢測事實及理由 (支援各種空白字符組合)
            elif normalized_text in ["事實及理由", "事實和理由", "事實與理由"]:
                markers["facts_and_reasons"].append(line_num)
                logger.debug(
                    "Detected 'facts_and_reasons' marker at line %d: %s",
                    line_num + 1,
                    line_text,
                )

            # 檢測附錄 - 限制條件：
            # 1. 必須在Date2之後（如果有Date2的話）
            # 2. 附錄關鍵詞必須是行首的前兩個字
            elif last_date_line is not None and line_num > last_date_line:
                # 檢查行首是否以附錄關鍵詞開始（前兩個字）
                if len(normalized_text) >= 2:
                    first_two_chars = normalized_text[:2]
                    if first_two_chars in ["附錄", "附件", "附圖", "附表"]:
                        markers["appendix"].append(line_num)
                        logger.debug(
                            "Detected 'appendix' marker at line %d (after Date2): %s",
                            line_num + 1,
                            line_text,
                        )

        return markers

    def create_line_based_chunks(
        self,
        lines: List[str],
        detection_results: List[HybridDetectionResult],
        learned_rules: List[LevelingRule],
    ) -> List[LineBasedChunk]:
        """基於行的分塊方法：
        1. 檢測特殊標記：主文、理由、事實、事實及理由、日期
        2. header (lv -3): 主文之前的行
        3. footer (lv -3): 最後日期之後的行
        4. content (lv -1): 兩個層級符號行之間的內容
        5. leveling_symbol (lv 1,2,3...): 檢測到的層級符號行
        """
        logger.debug("Starting line-based chunking.")

        # 步驟1: 檢測特殊標記
        special_markers = self.detect_special_markers(lines)

        # 步驟2: 建立規則映射 (從學習階段獲得)
        level_mapping = {}
        for rule in learned_rules:
            level_mapping[rule.symbol_category] = rule.assigned_level

        # 收集所有特殊標記行（包含日期）
        special_line_set = set()
        for marker_lines in special_markers.values():
            special_line_set.update(marker_lines)

        # 確定署名區域範圍（兩個日期之間）
        signature_line_set = set()
        date_lines_sorted = sorted(special_markers["dates"]) if special_markers["dates"] else []
        if len(date_lines_sorted) >= 2:
            first_date = date_lines_sorted[0]
            last_date = date_lines_sorted[-1]
            signature_line_set = set(range(first_date + 1, last_date))

        def emit_content_segment(segment_indices: List[int]):
            if not segment_indices:
                return
            segment_lines = [lines[idx] for idx in segment_indices]
            chunks.append(
                LineBasedChunk(
                    level=-1,
                    start_line=segment_indices[0],
                    end_line=segment_indices[-1],
                    chunk_type="content",
                    content_lines=segment_lines,
                    chunk_id=f"content_{segment_indices[0]}_{segment_indices[-1]}",
                )
            )

        def collect_content_segments(start_idx: int, end_idx: int):
            current_indices: List[int] = []
            for idx in range(start_idx, end_idx):
                # 跳過特殊標記、層級符號行和署名區域
                if idx in special_line_set or idx in leveling_symbol_lines or idx in signature_line_set:
                    if current_indices:
                        emit_content_segment(current_indices)
                        current_indices = []
                    continue
                current_indices.append(idx)
            if current_indices:
                emit_content_segment(current_indices)

        # 步驟3: 標記所有層級符號行
        leveling_symbol_lines = {}  # line_number -> (symbol, category, level)
        for result in detection_results:
            if result.final_prediction:
                symbol_category = result.symbol_category
                assigned_level = level_mapping.get(symbol_category, 1)  # 預設層級1
                leveling_symbol_lines[result.line_number - 1] = (
                    result.detected_symbol,
                    symbol_category,
                    assigned_level,
                )

        # 步驟4: 確定關鍵分界點
        # 找到主文位置 (Lv 0)
        main_text_line = (
            special_markers["main_text"][0] if special_markers["main_text"] else None
        )

        # 找到所有日期位置並確定 date1 和 date2
        date_lines = sorted(special_markers["dates"]) if special_markers["dates"] else []
        first_date_line = date_lines[0] if date_lines else None
        last_date_line = date_lines[-1] if len(date_lines) > 0 else None
        
        # 步驟5: 建立分塊
        chunks = []

        # Header區域 (Lv -3): 主文之前
        if main_text_line is not None and main_text_line > 0:
            header_content = lines[:main_text_line]
            chunks.append(
                LineBasedChunk(
                    level=-3,
                    start_line=0,
                    end_line=main_text_line - 1,
                    chunk_type="header",
                    content_lines=header_content,
                    chunk_id="header",
                )
            )
            logger.debug("Header segment captured (lines 1-%d).", main_text_line)

        # 確定內容區域的結束點
        content_end = len(lines) - 1

        # 如果有附錄標記，內容區域應該在第一個附錄標記之前結束
        first_appendix_line = (
            min(special_markers["appendix"]) if special_markers["appendix"] else None
        )
        if first_appendix_line is not None:
            content_end = first_appendix_line - 1

        # 處理主要內容區域
        content_start = main_text_line if main_text_line is not None else 0

        # 處理第一個日期 (Date1)
        if first_date_line is not None and content_start <= first_date_line <= content_end:
            chunks.append(
                LineBasedChunk(
                    level=-2,
                    start_line=first_date_line,
                    end_line=first_date_line,
                    chunk_type="date",
                    content_lines=[lines[first_date_line]],
                    chunk_id=f"date_{first_date_line}",
                )
            )
            logger.debug("Date1 marker at line %d.", first_date_line + 1)

        # 處理兩個日期之間的署名區域 (Signature) - 類似 header 處理，保留所有行
        if (first_date_line is not None and last_date_line is not None and 
            last_date_line > first_date_line and 
            content_start <= first_date_line <= content_end):
            
            signature_start = first_date_line + 1
            signature_end = last_date_line - 1
            
            if signature_start <= signature_end:
                # 保留所有行，不進行任何過濾（類似 header/footer 處理）
                signature_lines = [lines[line_idx] for line_idx in range(signature_start, signature_end + 1)]
                
                if signature_lines:
                    chunks.append(
                        LineBasedChunk(
                            level=-3,
                            start_line=signature_start,
                            end_line=signature_end,
                            chunk_type="signature",
                            content_lines=signature_lines,
                            chunk_id="signature",
                        )
                    )
                    logger.debug("Signature segment captured between Date1 and Date2 (lines %d-%d).", 
                               signature_start + 1, signature_end + 1)

        # 處理第二個日期 (Date2)
        if (last_date_line is not None and last_date_line != first_date_line and 
            content_start <= last_date_line <= content_end):
            chunks.append(
                LineBasedChunk(
                    level=-2,
                    start_line=last_date_line,
                    end_line=last_date_line,
                    chunk_type="date",
                    content_lines=[lines[last_date_line]],
                    chunk_id=f"date_{last_date_line}",
                )
            )
            logger.debug("Date2 marker at line %d.", last_date_line + 1)

        # 處理其他特殊標記 (主文、事實、理由、附錄等)
        for marker_type, line_numbers in special_markers.items():
            if marker_type == "dates":  # 日期已經單獨處理
                continue
            for line_num in line_numbers:
                if content_start <= line_num <= content_end:
                    chunks.append(
                        LineBasedChunk(
                            level=0,
                            start_line=line_num,
                            end_line=line_num,
                            chunk_type=marker_type,
                            content_lines=[lines[line_num]],
                            chunk_id=f"{marker_type}_{line_num}",
                        )
                    )

        # 內容區域分塊：根據層級符號行分割
        sorted_symbol_lines = sorted(leveling_symbol_lines.keys())

        current_pos = content_start
        for symbol_line in sorted_symbol_lines:
            if symbol_line < content_start or symbol_line > content_end:
                continue

            # 層級符號行之前的內容 (Lv -1)
            if current_pos < symbol_line:
                collect_content_segments(current_pos, symbol_line)

            # 層級符號行本身
            symbol, category, level = leveling_symbol_lines[symbol_line]
            chunks.append(
                LineBasedChunk(
                    level=level,
                    start_line=symbol_line,
                    end_line=symbol_line,
                    chunk_type="leveling_symbol",
                    content_lines=[lines[symbol_line]],
                    leveling_symbol=symbol,
                    chunk_id=f"level_{level}_{symbol_line}",
                )
            )

            current_pos = symbol_line + 1

        # 最後一個層級符號後的內容
        if current_pos <= content_end:
            collect_content_segments(current_pos, content_end + 1)

        # 處理附錄區域（如果有的話）
        if first_appendix_line is not None:
            appendix_start = first_appendix_line
            appendix_end = len(lines) - 1

            # 找到下一個主要結構標記（如果有的話）
            next_major_marker = None
            for marker_type, line_numbers in special_markers.items():
                if marker_type not in ["appendix", "dates"]:
                    for line_num in line_numbers:
                        if line_num > first_appendix_line:
                            next_major_marker = (
                                min(next_major_marker, line_num)
                                if next_major_marker
                                else line_num
                            )

            if next_major_marker:
                appendix_end = next_major_marker - 1

            # 處理附錄區域的內容
            appendix_current_pos = appendix_start
            for appendix_line in special_markers["appendix"]:
                if appendix_line < appendix_start or appendix_line > appendix_end:
                    continue

                # 附錄標記行之前的內容
                if appendix_current_pos < appendix_line:
                    appendix_content_indices = []
                    for idx in range(appendix_current_pos, appendix_line):
                        if (
                            idx not in special_line_set
                            and idx not in leveling_symbol_lines
                        ):
                            appendix_content_indices.append(idx)

                    if appendix_content_indices:
                        appendix_content_lines = [
                            lines[idx] for idx in appendix_content_indices
                        ]
                        chunks.append(
                            LineBasedChunk(
                                level=-1,
                                start_line=appendix_content_indices[0],
                                end_line=appendix_content_indices[-1],
                                chunk_type="content",
                                content_lines=appendix_content_lines,
                                chunk_id=f"content_{appendix_content_indices[0]}_{appendix_content_indices[-1]}",
                            )
                        )

                # 附錄標記行本身
                chunks.append(
                    LineBasedChunk(
                        level=0,
                        start_line=appendix_line,
                        end_line=appendix_line,
                        chunk_type="appendix",
                        content_lines=[lines[appendix_line]],
                        chunk_id=f"appendix_{appendix_line}",
                    )
                )

                appendix_current_pos = appendix_line + 1

            # 最後一個附錄標記後的內容
            if appendix_current_pos <= appendix_end:
                appendix_content_indices = []
                for idx in range(appendix_current_pos, appendix_end + 1):
                    if idx not in special_line_set and idx not in leveling_symbol_lines:
                        appendix_content_indices.append(idx)

                if appendix_content_indices:
                    appendix_content_lines = [
                        lines[idx] for idx in appendix_content_indices
                    ]
                    chunks.append(
                        LineBasedChunk(
                            level=-1,
                            start_line=appendix_content_indices[0],
                            end_line=appendix_content_indices[-1],
                            chunk_type="content",
                            content_lines=appendix_content_lines,
                            chunk_id=f"content_{appendix_content_indices[0]}_{appendix_content_indices[-1]}",
                        )
                    )

        # Footer區域 (Lv -3): Date2之後，且在附錄區域之前的內容
        # 注意：Date1和Date2之間的內容已經被處理為signature區域
        if last_date_line is not None:
            footer_start = last_date_line + 1
            footer_end = (
                first_appendix_line - 1 if first_appendix_line else len(lines) - 1
            )

            if footer_start <= footer_end:
                footer_lines = []
                for line_idx in range(footer_start, footer_end + 1):
                    if (
                        line_idx not in special_line_set
                        and line_idx not in leveling_symbol_lines
                        and line_idx not in signature_line_set
                    ):
                        footer_lines.append(lines[line_idx])

                if footer_lines:
                    chunks.append(
                        LineBasedChunk(
                            level=-3,
                            start_line=footer_start,
                            end_line=footer_end,
                            chunk_type="footer",
                            content_lines=footer_lines,
                            chunk_id="footer",
                        )
                    )
                    logger.debug(
                        "Footer segment captured (lines %d-%d), excluded special markers and appendix content.",
                        footer_start + 1,
                        footer_end + 1,
                    )

        # 按行號排序
        chunks.sort(key=lambda x: x.start_line)

        logger.info("Constructed %d line-based chunks.", len(chunks))

        # 統計分塊類型
        chunk_stats = {}
        for chunk in chunks:
            level_type = f"Lv{chunk.level}_{chunk.chunk_type}"
            chunk_stats[level_type] = chunk_stats.get(level_type, 0) + 1

        logger.debug("Chunk statistics:")
        for level_type, count in sorted(chunk_stats.items()):
            logger.debug("   %s: %d", level_type, count)

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

    def build_machine_tree(self, chunks: List[LineBasedChunk]) -> List[Dict[str, Any]]:
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
                recipient: Dict[str, Any] | None = None
                for candidate in reversed(stack):
                    if candidate["level"] >= 1:
                        recipient = candidate
                        break

                if recipient is not None:
                    recipient["content_lines"].extend(chunk.content_lines)
                    recipient["end_line"] = max(
                        recipient["end_line"], chunk.end_line + 1
                    )
                else:
                    tree.append(self._chunk_to_machine_node(chunk))
            else:
                # Header/Footer/Date 等保留為獨立節點
                tree.append(self._chunk_to_machine_node(chunk))

        return tree

    @staticmethod
    def _clean_text_for_rag(text: str) -> str:
        """清理文本用於 RAG：移除所有換行符和多餘空白"""
        # 移除所有換行符``
        text = text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\n\n", "").replace(" ", "")
        # 移除多餘的空白（將多個空白合併為一個）
        import re
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空白
        return text.strip()
    
    def build_rag_chunks(self, chunks: List[LineBasedChunk]) -> List[Dict[str, Any]]:
        """建立適合 RAG 檢索的分塊結構
        
        將層級符號行（level >= 1）與其管轄的內容（level -1）整合成完整的 chunk。
        每個 chunk 包含：
        - 層級符號標題（如果有）
        - 該層級下的所有內容
        - 完整的上下文路徑（從最頂層到當前層級）
        """
        if not chunks:
            return []
        
        logger.debug("Building RAG-ready chunks from line-based chunks.")
        
        # 按行號排序
        ordered_chunks = sorted(chunks, key=lambda x: x.start_line)
        
        # 建立層級結構
        rag_chunks = []
        level_stack = []  # 存儲當前的層級路徑
        current_chunk = None
        
        for chunk in ordered_chunks:
            # Header, Footer, Signature, Date 等特殊塊保持獨立
            if chunk.level < 0 and chunk.chunk_type in ["header", "footer", "signature", "date"]:
                # 先保存當前正在累積的 chunk（如果有）
                if current_chunk:
                    rag_chunks.append(current_chunk)
                    current_chunk = None
                
                content_text = "\n".join(line.replace("\r\n", "\n").replace("\r", "\n") for line in chunk.content_lines)
                full_text_cleaned = self._clean_text_for_rag(content_text)
                rag_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "level": chunk.level,
                    "start_line": chunk.start_line + 1,
                    "end_line": chunk.end_line + 1,
                    "title": chunk.chunk_type.upper(),
                    "content": content_text,
                    "full_text": full_text_cleaned,
                    "hierarchy_path": [],
                    "parent_titles": []
                })
                continue
            
            # Level 0 標記（主文、事實、理由等）- 保持獨立但記錄層級路徑
            if chunk.level == 0:
                # 先保存當前正在累積的 chunk（如果有）
                if current_chunk:
                    rag_chunks.append(current_chunk)
                    current_chunk = None
                
                # 清空層級堆疊，這些是主要章節標記
                title_text = chunk.content_lines[0].replace("\r\n", "\n").replace("\r", "\n").strip() if chunk.content_lines else chunk.chunk_type
                content_text = "\n".join(line.replace("\r\n", "\n").replace("\r", "\n") for line in chunk.content_lines)
                full_text_cleaned = self._clean_text_for_rag(content_text)
                
                level_stack = [{
                    "level": 0,
                    "title": title_text,
                    "start_line": chunk.start_line
                }]
                
                rag_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "level": chunk.level,
                    "start_line": chunk.start_line + 1,
                    "end_line": chunk.end_line + 1,
                    "title": title_text,
                    "content": content_text,
                    "full_text": full_text_cleaned,
                    "hierarchy_path": [],
                    "parent_titles": []
                })
                current_chunk = None
                continue
            
            # Level >= 1 層級符號行 - 開始新的 chunk
            if chunk.level >= 1 and chunk.chunk_type == "leveling_symbol":
                # 如果有正在處理的 chunk，先保存它
                if current_chunk:
                    # 清理 full_text：移除所有換行符和多餘空白
                    current_chunk["full_text"] = self._clean_text_for_rag(current_chunk["full_text"])
                    rag_chunks.append(current_chunk)
                
                # 更新層級堆疊
                while level_stack and level_stack[-1]["level"] >= chunk.level:
                    level_stack.pop()
                
                # 建立層級路徑
                hierarchy_path = [item["level"] for item in level_stack]
                parent_titles = [item["title"] for item in level_stack]
                
                # 創建新的 chunk - 移除 \r\n
                title = chunk.content_lines[0].replace("\r\n","").replace("\r","").replace("\n","").strip() if chunk.content_lines else f"Level {chunk.level}"
                
                current_chunk = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": "hierarchical_section",
                    "level": chunk.level,
                    "start_line": chunk.start_line + 1,
                    "end_line": chunk.end_line + 1,
                    "title": title,
                    "leveling_symbol": chunk.leveling_symbol,
                    "content": "",  # 將在後面添加
                    "full_text": title,  # 完整文本包含標題
                    "hierarchy_path": hierarchy_path,
                    "parent_titles": parent_titles
                }
                
                # 加入層級堆疊
                level_stack.append({
                    "level": chunk.level,
                    "title": title,
                    "start_line": chunk.start_line
                })
                
            # Level -1 內容 - 附加到當前 chunk
            elif chunk.level == -1 and chunk.chunk_type == "content":
                if current_chunk:
                    # 將內容添加到當前 chunk - 移除 \r\n
                    content_text = "\n".join(line.replace("\r\n", "\n").replace("\r", "\n") for line in chunk.content_lines)
                    if current_chunk["content"]:
                        current_chunk["content"] += "\n" + content_text
                        current_chunk["full_text"] += "\n" + content_text
                    else:
                        current_chunk["content"] = content_text
                        current_chunk["full_text"] += "\n" + content_text
                    
                    # 更新結束行
                    current_chunk["end_line"] = chunk.end_line + 1
                else:
                    # 沒有當前 chunk，可能是孤立的內容
                    # 創建一個獨立的內容 chunk - 移除 \r\n
                    content_text = "\n".join(line.replace("\r\n", "\n").replace("\r", "\n") for line in chunk.content_lines)
                    full_text_cleaned = self._clean_text_for_rag(content_text)
                    rag_chunks.append({
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": "orphan_content",
                        "level": -1,
                        "start_line": chunk.start_line + 1,
                        "end_line": chunk.end_line + 1,
                        "title": "Content",
                        "content": content_text,
                        "full_text": full_text_cleaned,
                        "hierarchy_path": [item["level"] for item in level_stack],
                        "parent_titles": [item["title"] for item in level_stack]
                    })
        
        # 保存最後一個 chunk
        if current_chunk:
            # 清理 full_text：移除所有換行符和多餘空白
            current_chunk["full_text"] = self._clean_text_for_rag(current_chunk["full_text"])
            rag_chunks.append(current_chunk)
        
        logger.info("Generated %d RAG-ready chunks.", len(rag_chunks))
        
        # 統計 chunk 類型
        chunk_type_stats = {}
        for chunk in rag_chunks:
            chunk_type = chunk["chunk_type"]
            chunk_type_stats[chunk_type] = chunk_type_stats.get(chunk_type, 0) + 1
        
        logger.debug("RAG chunk statistics:")
        for chunk_type, count in sorted(chunk_type_stats.items()):
            logger.debug("   %s: %d", chunk_type, count)
        
        return rag_chunks

    def build_machine_payload(self, result: AdaptiveDetectionResult) -> Dict[str, Any]:
        """組裝機器可讀輸出所需的 payload。"""
        rag_chunks = self.build_rag_chunks(result.line_based_chunks or [])
        
        return {
            "filename": result.filename,
            "learning_region": result.learning_region,
            "processing_stats": {
                "total_lines": result.processing_stats["total_lines"],
                "total_symbols_detected": result.processing_stats["total_symbols_detected"],
                "learned_rules_count": result.processing_stats["learned_rules_count"],
                "final_levels": result.processing_stats["final_levels"],
                "rag_chunks_count": len(rag_chunks)
            },
            "rag_chunks": rag_chunks
        }

    def export_machine_result(
        self, result: AdaptiveDetectionResult, output_dir: Path
    ) -> Path:
        """輸出機器可讀 JSON 檔，將 Lv -1 內容附加於上層層級 (排除 Lv 0)。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.build_machine_payload(result)

        target_name = f"{Path(result.filename).stem}_machine.json"
        target_path = output_dir / target_name
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target_path

    def concatenate_level_content(
        self, chunks: List[LineBasedChunk]
    ) -> Dict[str, List[str]]:
        """合併相同層級的內容 (步驟5: Concat the content of Lv -1 between lv 0 1 2 3 4 and so on)"""
        logger.debug("Merging content for identical levels.")

        level_content = {}

        for chunk in chunks:
            level_key = f"Lv_{chunk.level}"
            if level_key not in level_content:
                level_content[level_key] = []

            # 合併內容，並記錄分塊信息
            chunk_info = (
                f"[{chunk.chunk_type}:{chunk.start_line + 1}-{chunk.end_line + 1}]"
            )
            level_content[level_key].append(chunk_info)
            level_content[level_key].extend(chunk.content_lines)

        # 顯示合併結果統計
        logger.debug("Merged content statistics:")
        for level, content in sorted(level_content.items()):
            line_count = len([line for line in content if not line.startswith("[")])
            logger.debug("   %s: %d lines", level, line_count)

        return level_content

    def analyze_file_structure(self, file_path: Path) -> Tuple[bool, Dict]:
        """分析檔案結構 - 使用 comprehensive_analysis 的邏輯"""
        try:
            # 使用 judgment_splitter 處理單個檔案
            success, result = process_single_file(file_path)

            if not success:
                return False, {}

            # 讀取原始數據
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 分析章節結構 - result 是字典
            sections = result.get("sections", {}) if isinstance(result, dict) else {}
            has_main_text = bool(sections.get("main_text", []))
            has_facts = bool(sections.get("facts", []))
            has_reasons = bool(sections.get("reasons", []))
            has_facts_and_reasons = bool(sections.get("facts_and_reasons", []))

            # 確定學習區間類型
            learning_region = None
            learning_lines = []

            if has_facts_and_reasons:
                # S-D 區間：從 facts_and_reasons 到文件末尾
                learning_region = "S-D"
                fr_lines = sections.get("facts_and_reasons", [])
                if fr_lines:
                    # 獲取 facts_and_reasons 開始的行號
                    full_lines = data["JFULL"].split("\n")
                    fr_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [
                            l.strip() for l in fr_lines[:3]
                        ]:
                            fr_start_line = i
                            break

                    if fr_start_line is not None:
                        learning_lines = full_lines[fr_start_line:]

            elif has_reasons:
                # R-D 區間：從 reasons 到文件末尾
                learning_region = "R-D"
                reasons_lines = sections.get("reasons", [])
                if reasons_lines:
                    full_lines = data["JFULL"].split("\n")
                    reasons_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [
                            l.strip() for l in reasons_lines[:3]
                        ]:
                            reasons_start_line = i
                            break

                    if reasons_start_line is not None:
                        learning_lines = full_lines[reasons_start_line:]

            if not learning_region:
                # 沒有 R 或 S 章節，使用全文
                learning_region = "全文"
                learning_lines = data["JFULL"].split("\n")

            structure_info = {
                "sections": sections,
                "has_main_text": has_main_text,
                "has_facts": has_facts,
                "has_reasons": has_reasons,
                "has_facts_and_reasons": has_facts_and_reasons,
                "learning_region": learning_region,
                "learning_lines": learning_lines,
                "full_text_lines": data["JFULL"].split("\n"),
                "total_lines": len(data["JFULL"].split("\n")),
            }

            return True, structure_info

        except Exception as exc:
            logger.exception("Failed to analyze file structure for %s", file_path)
            return False, {}

    def learn_leveling_rules(
        self, learning_lines: List[str], learning_region: str, verbose: bool = False
    ) -> List[LevelingRule]:
        """在學習區間建立層級規則 - 完全動態學習

        不再依賴任何預定義層級，完全基於文件本身的符號出現順序
        """
        logger.info("Learning hierarchy rules within region %s.", learning_region)
        logger.debug("Learning span contains %d lines.", len(learning_lines))

        # 在學習區間執行檢測
        learning_results = self.hybrid_detector.detect_hybrid_markers(
            learning_lines, verbose=verbose
        )

        # 獲取學習區間的層級分析
        self.hybrid_detector.detection_results = learning_results
        hierarchy_analysis = self.hybrid_detector.detect_hierarchy_levels()

        if not hierarchy_analysis or not hierarchy_analysis.get("level_mapping"):
            logger.warning("No valid hierarchy rules found in the learning region.")
            return []

        # 建立規則 - 完全基於學習的層級
        rules = []
        level_mapping = hierarchy_analysis["level_mapping"]

        logger.info(
            "Learned hierarchy rules for %d symbol categories.", len(level_mapping)
        )

        for symbol_category, level_info in level_mapping.items():
            rule = LevelingRule(
                symbol_category=symbol_category,
                assigned_level=level_info["assigned_level"],
                confidence=level_info["count"]
                / len([r for r in learning_results if r.final_prediction]),
                learning_source=learning_region,
                occurrences=level_info["count"],
                examples=[ex["text"][:50] + "..." for ex in level_info["examples"][:3]],
            )
            rules.append(rule)

            logger.debug(
                "   %s -> level %d (confidence %.3f)",
                symbol_category,
                rule.assigned_level,
                rule.confidence,
            )

        return rules

    def apply_leveling_rules(
        self,
        full_results: List[HybridDetectionResult],
        learned_rules: List[LevelingRule],
    ) -> Dict:
        """將學習到的規則應用到全文檢測結果"""
        logger.debug("Applying learned hierarchy rules to the full document.")

        # 建立規則映射
        rule_mapping = {}
        for rule in learned_rules:
            rule_mapping[rule.symbol_category] = rule.assigned_level

        # 應用規則到全文結果
        enhanced_hierarchy = []
        unknown_categories = set()
        next_available_level = max(rule_mapping.values()) + 1 if rule_mapping else 1

        for result in full_results:
            if not result.final_prediction:
                continue

            symbol_category = result.symbol_category

            if symbol_category in rule_mapping:
                # 使用學習到的規則
                assigned_level = rule_mapping[symbol_category]
            else:
                # 新的符號類型，分配新層級
                if symbol_category not in unknown_categories:
                    rule_mapping[symbol_category] = next_available_level
                    unknown_categories.add(symbol_category)
                    next_available_level += 1

                assigned_level = rule_mapping[symbol_category]

            enhanced_hierarchy.append(
                {
                    "line_number": result.line_number,
                    "detected_symbol": result.detected_symbol,
                    "symbol_category": symbol_category,
                    "assigned_level": assigned_level,
                    "is_learned_rule": symbol_category not in unknown_categories,
                    "line_text": result.line_text,
                    "method_used": result.method_used,
                    "bert_score": result.bert_score,
                }
            )

        # 創建層級映射統計
        level_stats = {}
        for item in enhanced_hierarchy:
            category = item["symbol_category"]
            if category not in level_stats:
                level_stats[category] = {
                    "assigned_level": item["assigned_level"],
                    "count": 0,
                    "is_learned": item["is_learned_rule"],
                    "examples": [],
                }
            level_stats[category]["count"] += 1
            if len(level_stats[category]["examples"]) < 3:
                level_stats[category]["examples"].append(
                    {
                        "line": item["line_number"],
                        "symbol": item["detected_symbol"],
                        "text": item["line_text"][:50] + "...",
                    }
                )

        logger.info("Rule application completed:")
        logger.info(
            "   Known rules: %d types", len(rule_mapping) - len(unknown_categories)
        )
        logger.info("   New discoveries: %d types", len(unknown_categories))
        logger.info("   Total hierarchy symbols: %d", len(enhanced_hierarchy))

        return {
            "enhanced_hierarchy": enhanced_hierarchy,
            "level_mapping": level_stats,
            "rule_coverage": (len(rule_mapping) - len(unknown_categories))
            / len(rule_mapping)
            if rule_mapping
            else 0,
            "total_levels": len(
                set(item["assigned_level"] for item in enhanced_hierarchy)
            ),
            "total_symbols": len(enhanced_hierarchy),
        }

    def process_single_file(
        self, file_path: Path, verbose: bool = False
    ) -> Optional[AdaptiveDetectionResult]:
        """處理單個檔案 - 完整的自適應檢測流程 + 基於行的分塊"""
        try:
            if verbose:
                logger.info("Processing file: %s", file_path.name)

            # 步驟1: 文件分塊
            success, structure_info = self.analyze_file_structure(file_path)
            if not success:
                if verbose:
                    logger.error("File structure analysis failed for %s", file_path)
                return None

            learning_region = structure_info["learning_region"]
            if verbose:
                logger.info("Detected document structure region: %s", learning_region)

            # 步驟2: 全文層級符號偵測
            if verbose:
                logger.debug("Running full-document hierarchy detection.")
            full_text_lines = structure_info["full_text_lines"]
            full_detection_results = self.hybrid_detector.detect_hybrid_markers(
                full_text_lines, verbose=verbose
            )

            # 步驟3: 規則學習區間
            learning_lines = structure_info["learning_lines"]
            learned_rules = self.learn_leveling_rules(
                learning_lines, learning_region, verbose=verbose
            )

            # 步驟4: 層級規則建立與全文應用
            applied_hierarchy = self.apply_leveling_rules(
                full_detection_results, learned_rules
            )

            # 步驟5: 基於行的分塊 (新增)
            if verbose:
                logger.debug("Executing line-based chunk generation.")
            line_based_chunks = self.create_line_based_chunks(
                full_text_lines, full_detection_results, learned_rules
            )

            # 步驟6: 合併相同層級內容 (新增)
            # 步驟6: 合併相同層級內容 (新增)
            level_content = self.concatenate_level_content(line_based_chunks)

            # 處理統計
            processing_stats = {
                "total_lines": structure_info["total_lines"],
                "learning_lines": len(learning_lines),
                "total_symbols_detected": len(
                    [r for r in full_detection_results if r.final_prediction]
                ),
                "learned_rules_count": len(learned_rules),
                "rule_coverage": applied_hierarchy["rule_coverage"],
                "final_levels": applied_hierarchy["total_levels"],
                "line_based_chunks_count": len(line_based_chunks),  # 新增
                "level_content_summary": {
                    k: len([line for line in v if not line.startswith("[")])
                    for k, v in level_content.items()
                },  # 新增
            }

            result = AdaptiveDetectionResult(
                filename=file_path.name,
                file_structure=structure_info,
                learning_region=learning_region,
                learned_rules=learned_rules,
                full_detection_results=full_detection_results,
                applied_hierarchy=applied_hierarchy,
                processing_stats=processing_stats,
                line_based_chunks=line_based_chunks,  # 新增
            )

            return result
        
        finally:
            # 清理 CUDA 快取以避免 OOM
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_sample_directory(
        self,
        sample_dir: Path,
        output_dir: Optional[Path] = None,
        max_files: Optional[int] = None,
        verbose: bool = False,
        generate_reports: bool = True,
    ):
        """處理 sample 目錄中的所有檔案"""
        if verbose:
            logger.info(
                "Starting batch adaptive detection for directory: %s", sample_dir
            )

        if not sample_dir.exists():
            logger.error("Directory does not exist: %s", sample_dir)
            return

        json_files = sorted(p for p in sample_dir.glob("*.json") if p.is_file())
        if max_files is not None:
            json_files = json_files[:max_files]
        if not json_files:
            if verbose:
                logger.warning("No JSON files found under %s", sample_dir)
            return

        if verbose:
            logger.info("Found %d JSON file(s) to process.", len(json_files))

        all_results = []
        learning_region_stats = {"S-D": 0, "R-D": 0, "全文": 0}
        exported_files = []  # Track exported machine-readable files

        # Simple progress bar
        print(f"Processing {len(json_files)} files...")

        for i, json_file in enumerate(json_files, 1):
            if not verbose:
                # Simple progress indicator
                progress = f"[{i}/{len(json_files)}] {json_file.name}"
                print(f"\r{progress:<60}", end="", flush=True)
            else:
                logger.info(
                    "Processing file %d/%d: %s", i, len(json_files), json_file.name
                )

            result = self.process_single_file(json_file, verbose=verbose)
            if result:
                all_results.append(result)
                learning_region_stats[result.learning_region] += 1

                # Export individual machine-readable result for each file
                try:
                    export_path = self.export_machine_result(
                        result, output_dir or Path("output")
                    )
                    exported_files.append(export_path)
                    if verbose:
                        logger.debug(
                            "Exported machine result for %s to %s",
                            json_file.name,
                            export_path,
                        )
                except Exception as exc:
                    if verbose:
                        logger.error(
                            "Failed to export machine result for %s: %s",
                            json_file.name,
                            exc,
                        )
            else:
                if verbose:
                    logger.error("Processing failed for %s", json_file.name)

        if not verbose:
            print()  # New line after progress

        # 生成綜合報告
        if generate_reports:
            self.generate_batch_report(all_results, learning_region_stats, output_dir)

        if verbose:
            logger.info(
                "Successfully processed %d files and exported %d machine results",
                len(all_results),
                len(exported_files),
            )
        else:
            if generate_reports:
                print(
                    f"✅ Completed: {len(all_results)} files processed, {len(exported_files)} machine results exported"
                )
            else:
                print(
                    f"✅ Completed: {len(all_results)} files processed, {len(exported_files)} machine results exported"
                )

    def generate_batch_report(
        self,
        results: List[AdaptiveDetectionResult],
        region_stats: Dict,
        output_dir: Optional[Path] = None,
    ):
        """生成批量處理報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(output_dir) if output_dir else Path("output")
        target_dir.mkdir(parents=True, exist_ok=True)
        report_file = target_dir / f"adaptive_detection_report_{timestamp}.md"

        report = f"""# 自適應混合層級符號檢測報告
生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
處理檔案: {len(results)} 個

## 📊 整體統計

### 學習區間分布
- **S-D 區間** (事實理由合併): {region_stats["S-D"]} 檔案
- **R-D 區間** (理由章節): {region_stats["R-D"]} 檔案
- **全文檢測**: {region_stats["全文"]} 檔案

### 處理統計
"""

        if results:
            total_lines = sum(r.processing_stats["total_lines"] for r in results)
            total_symbols = sum(
                r.processing_stats["total_symbols_detected"] for r in results
            )
            avg_coverage = sum(
                r.processing_stats["rule_coverage"] for r in results
            ) / len(results)

            report += f"- 總行數: {total_lines:,}\n"
            report += f"- 總符號數: {total_symbols:,}\n"
            report += f"- 平均規則覆蓋率: {avg_coverage:.1%}\n"

            # 各檔案詳細結果
            report += "\n## 📋 各檔案檢測結果\n\n"

            for i, result in enumerate(results, 1):
                stats = result.processing_stats
                hierarchy = result.applied_hierarchy

                report += f"### {i}. {result.filename}\n"
                report += f"- **學習模式**: {result.learning_region}\n"
                report += f"- **總行數**: {stats['total_lines']:,}\n"
                report += f"- **學習範圍**: {stats['learning_lines']:,} 行\n"
                report += f"- **檢測符號**: {stats['total_symbols_detected']:,} 個\n"
                report += f"- **學習規則**: {stats['learned_rules_count']} 種\n"
                report += f"- **規則覆蓋**: {stats['rule_coverage']:.1%}\n"
                report += f"- **最終層級**: {stats['final_levels']} 層\n"
                report += (
                    f"- **基於行分塊**: {stats.get('line_based_chunks_count', 0)} 個\n"
                )

                # 顯示層級內容統計
                if "level_content_summary" in stats:
                    report += f"\n**層級內容統計:**\n"
                    for level, line_count in sorted(
                        stats["level_content_summary"].items()
                    ):
                        report += f"  - {level}: {line_count} 行\n"

                # 顯示學習到的規則
                if result.learned_rules:
                    report += f"\n**學習規則:**\n"
                    for rule in result.learned_rules[:5]:  # 只顯示前5個
                        report += f"  - {rule.symbol_category}: L{rule.assigned_level} (信心度: {rule.confidence:.3f})\n"

                # 顯示層級結構預覽
                if hierarchy.get("enhanced_hierarchy"):
                    report += f"\n**層級結構預覽:**\n"
                    for item in hierarchy["enhanced_hierarchy"][:5]:  # 只顯示前5個
                        learned_mark = "✓" if item["is_learned_rule"] else "✗"
                        indent = "  " * item["assigned_level"]
                        report += f"  {indent}L{item['assigned_level']} {learned_mark} 行{item['line_number']:4}: {item['detected_symbol']} - {item['line_text'][:40]}...\n"

                report += "\n"

        # 保存報告
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Adaptive detection report saved to %s", report_file)

        # 保存詳細數據
        json_file = target_dir / f"adaptive_detection_data_{timestamp}.json"
        json_data = []

        for result in results:
            # 轉換為可序列化的格式
            chunk_data = []
            if result.line_based_chunks:
                for chunk in result.line_based_chunks:
                    chunk_data.append(
                        {
                            "level": chunk.level,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "chunk_type": chunk.chunk_type,
                            "content_lines": list(chunk.content_lines),
                            "leveling_symbol": chunk.leveling_symbol,
                            "chunk_id": chunk.chunk_id,
                        }
                    )

            json_data.append(
                {
                    "filename": result.filename,
                    "learning_region": result.learning_region,
                    "processing_stats": result.processing_stats,
                    "learned_rules": [
                        {
                            "symbol_category": rule.symbol_category,
                            "assigned_level": rule.assigned_level,
                            "confidence": rule.confidence,
                            "learning_source": rule.learning_source,
                            "occurrences": rule.occurrences,
                            "examples": rule.examples,
                        }
                        for rule in result.learned_rules
                    ],
                    "applied_hierarchy": result.applied_hierarchy,
                    "line_based_chunks": chunk_data,  # 新增
                }
            )

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        logger.info("Detailed detection data saved to %s", json_file)


def main():
    """主函數 - 自適應檢測演示"""
    logger.info("Adaptive detector demo following the 'learn then apply' principle.")
    logger.info(
        "Workflow: document chunking -> rule learning -> document-wide application."
    )

    # 初始化自適應檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = AdaptiveHybridDetector(model_path if Path(model_path).exists() else None)

    # 處理 sample 目錄
    sample_dir = Path("data/processed/sample")
    detector.process_sample_directory(sample_dir, verbose=False)


if __name__ == "__main__":
    main()
