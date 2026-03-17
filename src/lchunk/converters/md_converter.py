#!/usr/bin/env python3
"""Markdown converter — 支援兩種輸入模式

Mode A（舊版，保留向後相容）:
    MarkdownConverter — 從 AdaptiveDetectionResult / AdaptiveHybridDetector 轉換

Mode B（新版 Pipeline）:
    PipelineMarkdownConverter — 從 JudgmentArtifact（hierarchy_tree 巢狀樹）轉換
    可接受:
      1. JudgmentArtifact 物件（直接由 PipelineOrchestrator.process_file() 返回）
      2. pipeline debug JSON（由 pipeline.py --save 輸出）
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────────────────
# Mode A: 舊版依賴（保持向後相容，匯入失敗時降級）
# ──────────────────────────────────────────────────────────
try:
    from ..detectors.adaptive_hierarchy import (
        AdaptiveDetectionResult,
        AdaptiveHybridDetector,
        LevelingRule,
        LineBasedChunk,
    )
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False


# ══════════════════════════════════════════════════════════
# Mode B: Pipeline Markdown Converter
# ══════════════════════════════════════════════════════════

class PipelineMarkdownConverter:
    """將 Pipeline 產出的 JudgmentArtifact（或其 debug JSON）轉換為 Markdown。

    JudgmentArtifact.hierarchy_tree 是一個巢狀樹，每個節點結構為：
        {
            "level": int,          # -3=header, -2=date, -1=content, 0=main_text/facts/reasons, ≥1=leveling_symbol
            "chunk_type": str,     # "header", "content", "main_text", "facts", "reasons", "leveling_symbol", ...
            "start_line": int,
            "end_line": int,
            "content_lines": List[str],
            "leveling_symbol": str | None,
            "chunk_id": str,
            "children": List[dict]
        }

    層級對映至 Markdown:
        level == -3          → 純文字（法院表頭）
        level == -2          → 斜體日期 *...*
        level == -1          → 段落（合併）
        level == 0           → ## 標題（主文 / 事實 / 理由 / …）
        level >= 1           → 縮排有序清單，深度 = level - 1
    """

    # ── 辨識特殊標題的靜態模式 ────────────────────────────────────
    _SECTION_TITLES = frozenset(['主文', '事實', '理由', '事實及理由', '主文及理由'])
    _APPENDIX_PAT = re.compile(r'^附[錄件圖表]')
    _DATE_PAT = re.compile(
        r'^(中\s*華\s*民\s*國.*\d+.*年.*\d+.*月.*\d+.*日|\d{3,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)$'
    )

    def __init__(self) -> None:
        pass

    # ── 公開 API ──────────────────────────────────────────────────

    def from_artifact(self, artifact: Any) -> str:
        """從 JudgmentArtifact 物件產生 Markdown 字串。"""
        filename = Path(artifact.file_path).stem if artifact.file_path else "文件"
        metadata = artifact.metadata or {}
        hierarchy_tree: List[Dict] = artifact.hierarchy_tree or []
        return self._render(filename, metadata, hierarchy_tree)

    def from_debug_json(self, json_path: Path) -> str:
        """從 pipeline.py --save 輸出的 debug JSON 產生 Markdown 字串。
        
        自動相容兩種格式：
          1. pipeline.py --save 輸出（含 hierarchy_tree）→ 直接渲染
          2. splitter_refactor.py --save 輸出（含 sections/full_lines，無 hierarchy_tree）
             → 從 sections 建立簡易 hierarchy_tree 再渲染
        同時自動遷移舊版格式（footer level -3 → date2 level -2）。
        """
        data = json.loads(json_path.read_text(encoding="utf-8"))
        filename = Path(data.get("metadata", {}).get("JID", json_path.stem)).stem
        metadata = data.get("metadata", {})
        hierarchy_tree: List[Dict] = data.get("hierarchy_tree", [])

        if not hierarchy_tree:
            # 嘗試從 sections / full_lines 建立簡易 hierarchy_tree
            hierarchy_tree = self._build_tree_from_sections(data)

        # 自動遷移舊版 footer/sig 格式
        hierarchy_tree = self._migrate_legacy_nodes(hierarchy_tree)
        return self._render(filename, metadata, hierarchy_tree)

    @staticmethod
    def _build_tree_from_sections(data: Dict) -> List[Dict]:
        """從 splitter_refactor 輸出的 sections 建立簡易 hierarchy_tree。

        sections 結構：
          { "header": { "name": "header", "lines": [{ "index", "original_text", ... }] }, ... }
        """
        sections_raw: Dict = data.get("sections", {})
        if not sections_raw:
            return []

        # section 順序決定輸出順序
        ORDER = ["header", "main_text", "facts", "reasons", "facts_and_reasons",
                 "date1", "sig", "date2", "appendix"]

        # 用於偵測附錄標頭
        _appendix_pat = re.compile(r'^附[錄件圖表]')

        def _lines_of(section_name: str) -> List[str]:
            sec = sections_raw.get(section_name, {})
            rows = sec.get("lines", []) if isinstance(sec, dict) else sec
            return [r.get("original_text", "") if isinstance(r, dict) else str(r) for r in rows]

        def _make_node(level: int, chunk_type: str, chunk_id: str, lines: List[str]) -> Dict:
            non_empty = [l for l in lines if l.strip()]
            if not non_empty:
                return None
            return {
                "level": level,
                "chunk_type": chunk_type,
                "start_line": 0,
                "end_line": len(lines),
                "content_lines": lines,
                "leveling_symbol": None,
                "chunk_id": chunk_id,
                "children": [],
            }

        nodes: List[Dict] = []

        # ── header ──────────────────────────────────────────────────
        n = _make_node(-3, "header", "header", _lines_of("header"))
        if n: nodes.append(n)

        # ── 主文、事實、理由（作為 H2 節）──────────────────────────
        for sname, ctype in [("main_text", "main_text"),
                              ("facts", "facts"),
                              ("reasons", "reasons"),
                              ("facts_and_reasons", "facts_and_reasons")]:
            lines = _lines_of(sname)
            if not lines:
                continue
            # 第一行是 section 觸發關鍵字（如「主文」），作為 H2
            header_line = [lines[0]]
            body_lines  = lines[1:]
            n = _make_node(0, ctype, sname, header_line)
            if n:
                # 把正文內容作為 level -1 child
                if body_lines:
                    body_node = _make_node(-1, "content", f"{sname}_content", body_lines)
                    if body_node:
                        n["children"].append(body_node)
                nodes.append(n)

        # ── date1 ────────────────────────────────────────────────────
        n = _make_node(-2, "date1", "date1", _lines_of("date1"))
        if n: nodes.append(n)

        # ── sig ───────────────────────────────────────────────────────
        n = _make_node(-2, "sig", "sig", _lines_of("sig"))
        if n: nodes.append(n)

        # ── date2 ────────────────────────────────────────────────────
        n = _make_node(-2, "date2", "date2", _lines_of("date2"))
        if n: nodes.append(n)

        # ── appendix：按附錄標頭拆分 ─────────────────────────────────
        appendix_lines = _lines_of("appendix")
        if appendix_lines:
            current: List[str] = []
            idx = 0

            def _flush_appendix():
                nonlocal current, idx
                if current:
                    n = _make_node(0, "appendix", f"appendix_{idx}", current)
                    if n: nodes.append(n)
                    idx += 1
                    current = []

            for line in appendix_lines:
                cleaned = re.sub(r'[\s\u3000]+', '', line)
                if _appendix_pat.match(cleaned) and current:
                    _flush_appendix()
                current.append(line)
            _flush_appendix()

        return nodes



    @staticmethod
    def _migrate_legacy_nodes(nodes: List[Dict]) -> List[Dict]:
        """將舊版格式的 hierarchy_tree 節點升級為新格式。

        舊版問題：
          - footer (level -3)：實際上是 date2（正本送達日期）
          - sig (level -2)：可能包含 date1 日期行在最前面

        新格式：
          - date2 (level -2, chunk_type='date2')
          - date1 (level -2, chunk_type='date1')
          - sig (level -2, chunk_type='sig')
        """
        import re
        # 日期偵測：相容全形空白（\u3000），直接 search 原始文字
        _date_pat = re.compile(
            r'中[\s\u3000]*華[\s\u3000]*民[\s\u3000]*國'
            r'[\s\u3000\d]+年[\s\u3000\d]+月[\s\u3000\d]+日',
            re.UNICODE
        )

        def _is_date_line(text: str) -> bool:
            return bool(_date_pat.search(text))


        migrated: List[Dict] = []
        for node in nodes:
            chunk_type = node.get("chunk_type", "")
            level = node.get("level", -1)
            content_lines: List[str] = node.get("content_lines", [])

            # ── 舊版 footer：level -3，實際是 date2 ────────────────
            if chunk_type == "footer" and level == -3:
                node = dict(node)  # 複製避免 mutate 原始資料
                node["level"] = -2
                node["chunk_type"] = "date2"
                node["chunk_id"] = "date2"
                migrated.append(node)
                continue

            # ── 舊版 sig：若第一行是日期，拆出 date1 ─────────────────
            if chunk_type == "sig" and level == -2 and content_lines:
                first_line = content_lines[0].strip()
                if _is_date_line(first_line):
                    # 第一行是 date1
                    migrated.append({
                        "level": -2,
                        "chunk_type": "date1",
                        "start_line": node.get("start_line"),
                        "end_line": node.get("start_line"),
                        "content_lines": [content_lines[0]],
                        "leveling_symbol": None,
                        "chunk_id": "date1",
                        "children": [],
                    })
                    # 剩餘行作為 sig
                    if len(content_lines) > 1:
                        node = dict(node)
                        node["content_lines"] = content_lines[1:]
                        migrated.append(node)
                    continue

            # ── 遞迴處理 children ─────────────────────────────────────
            if node.get("children"):
                node = dict(node)
                node["children"] = PipelineMarkdownConverter._migrate_legacy_nodes(node["children"])

            migrated.append(node)

        return migrated


    def convert_file(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        model_path: Optional[Path] = None,
        from_debug_json: bool = False,
    ) -> Path:
        """處理單一檔案並將 Markdown 寫入 output_dir，返回輸出路徑。

        Args:
            input_path:      輸入檔案路徑（原始 JSON 或 debug JSON）
            output_dir:      輸出目錄
            model_path:      BERT 模型路徑（當 from_debug_json=False 時使用）
            from_debug_json: True → 直接讀 debug JSON；False → 跑完整 Pipeline
        """
        if from_debug_json:
            md_text = self.from_debug_json(input_path)
        else:
            from src.lchunk.pipeline import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                model_path=str(model_path) if (model_path and model_path.exists()) else None
            )
            artifact = orchestrator.process_file(input_path)
            md_text = self.from_artifact(artifact)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{input_path.stem}.md"
        out_file.write_text(md_text, encoding="utf-8")
        return out_file

    def convert_batch(
        self,
        input_files: List[Path],
        output_dir: Path,
        *,
        model_path: Optional[Path] = None,
        from_debug_json: bool = False,
    ) -> List[Path]:
        """批次轉換多個檔案，返回成功寫出的 Markdown 路徑清單。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for src in input_files:
            try:
                out = self.convert_file(src, output_dir, model_path=model_path, from_debug_json=from_debug_json)
                written.append(out)
                print(f"  ✅ {src.name} → {out.name}")
            except Exception as exc:
                print(f"  ❌ 失敗 {src.name}: {exc}")
        return written

    # ── 核心渲染邏輯 ──────────────────────────────────────────────

    def _render(self, filename: str, metadata: Dict, nodes: List[Dict]) -> str:
        """遞迴渲染 hierarchy_tree 成 Markdown 字串。"""
        lines: List[str] = [f"# {filename}", ""]

        # 附加元資料摘要（法院 / 案號）
        court = metadata.get("court_full_name", "")
        jid   = metadata.get("JID", "")
        if court or jid:
            meta_parts = [p for p in [court, jid] if p]
            lines.append(f"> {' ／ '.join(meta_parts)}")
            lines.append("")

        self._render_nodes(nodes, lines, depth=0)

        # 收尾清理：去除連續多個空行
        result: List[str] = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            result.append(line.rstrip())
            prev_blank = is_blank

        return "\n".join(result).strip() + "\n"

    def _render_nodes(self, nodes: List[Dict], out: List[str], depth: int) -> None:
        """遞迴處理 nodes 列表（depth 記錄巢狀深度，用於縮排）。"""
        for node in nodes:
            self._render_node(node, out, depth)

    def _render_node(self, node: Dict, out: List[str], depth: int) -> None:
        level: int = node.get("level", -1)
        chunk_type: str = node.get("chunk_type", "content")
        content_lines: List[str] = node.get("content_lines", [])
        children: List[Dict] = node.get("children", [])

        # 過濾空行，合併成單一字串（也保留多行版本用於 header）
        non_empty = [ln.strip() for ln in content_lines if ln.strip()]
        joined = " ".join(non_empty)

        # ── level -3: 法院表頭 / 頁首頁尾 ────────────────────────
        if level == -3:
            if non_empty:
                out.extend(non_empty)
                out.append("")
            self._render_nodes(children, out, depth)
            return

        # ── level -2: date1 / sig / date2 ────────────────────────────
        if level == -2:
            if chunk_type == 'sig':
                # sig：保持逐行原文（法官姓名、書記官等結構不合并）
                if non_empty:
                    out.extend(non_empty)
                    out.append("")
            else:
                # date1 / date2（及其他 level -2）：斜體單行
                if joined:
                    out.append(f"*{joined}*")
                    out.append("")
            self._render_nodes(children, out, depth)
            return

        # ── level 0: 大節標題（主文、事實、理由…）→ H2 ───────────
        if level == 0:
            if chunk_type == 'appendix':
                # appendix chunk：首行為標頭（## 標題），其餘為段落內容
                if non_empty:
                    out.append(f"## {non_empty[0]}")
                    out.append("")
                    for body_line in non_empty[1:]:
                        out.append(body_line)
                    if len(non_empty) > 1:
                        out.append("")
            else:
                title = re.sub(r'\s+', '', joined)  # 去除全形空白
                if title:
                    out.append(f"## {title}")
                    out.append("")
                # 先渲染本節的 children，level 0 下通常是 level 1/2/3...
                self._render_nodes(children, out, depth)
            return


        # ── level -1: 一般段落（非層級符號） ─────────────────────
        if level == -1:
            if joined:
                out.append(joined)
                out.append("")
            self._render_nodes(children, out, depth)
            return

        # ── level >= 1: 層級符號條文 ──────────────────────────────
        if level >= 1:
            indent = "    " * (level - 1)
            # 取得層級符號前綴（若有）
            symbol = node.get("leveling_symbol") or ""
            if joined:
                out.append(f"{indent}- {joined}")
                out.append("")
            # 遞迴子節點（深度加一）
            self._render_nodes(children, out, depth + 1)
            return

        # ── 其他未知 level：當段落處理 ────────────────────────────
        if joined:
            out.append(joined)
            out.append("")
        self._render_nodes(children, out, depth)


# ══════════════════════════════════════════════════════════
# Mode A: 舊版 MarkdownConverter（保留向後相容）
# ══════════════════════════════════════════════════════════

class MarkdownConverter:
    """Convert adaptive detection results to Markdown format.

    Note: 此類別依賴已廢棄的 AdaptiveHybridDetector / AdaptiveDetectionResult。
    新專案請改用 PipelineMarkdownConverter。
    """

    def __init__(self, detector=None, model_path=None):
        """Initialize the converter with an optional detector instance or model path."""
        if not _LEGACY_AVAILABLE:
            raise ImportError(
                "MarkdownConverter 需要 AdaptiveHybridDetector，但匯入失敗。"
                "請改用 PipelineMarkdownConverter。"
            )
        if detector is not None:
            self.detector = detector
        elif model_path is not None:
            self.detector = AdaptiveHybridDetector(
                str(model_path) if Path(model_path).exists() else None
            )
        else:
            self.detector = None

    @staticmethod
    def sanitize_lines(lines: Iterable[str]) -> List[str]:
        """Clean and filter lines, removing empty or whitespace-only lines."""
        return [line.strip() for line in lines if line and line.strip()]

    @staticmethod
    def is_appendix_content(text: str) -> bool:
        """Check if text IS an appendix header (not just contains appendix keywords)."""
        text_cleaned = text.strip()
        if len(text_cleaned) > 100:
            return False
        appendix_patterns = [r'^附錄', r'^附件', r'^附圖', r'^附表']
        text_normalized = re.sub(r'\s+', '', text_cleaned)
        return (any(re.match(pattern, text_cleaned) for pattern in appendix_patterns) or
                any(re.match(pattern, text_normalized) for pattern in appendix_patterns))

    @staticmethod
    def is_standalone_appendix_header(text: str, is_in_footer: bool = False) -> bool:
        """Check if text is a standalone appendix header in footer section."""
        if not is_in_footer:
            return False
        standalone_patterns = [r'^附錄.*', r'^附件.*', r'^附圖.*', r'^附表[一二三四五六七八九十\\d]*$']
        return any(re.match(pattern, text.strip()) for pattern in standalone_patterns)

    @staticmethod
    def find_date2_position(sorted_chunks) -> int:
        """Find the position of Date2 section in the chunks."""
        for i, chunk in enumerate(sorted_chunks):
            content_text = " ".join(chunk.content_lines)
            if re.search(r'中\s*華\s*民\s*國.*年.*月.*日', content_text) or chunk.level == -2:
                return i
        return -1

    @staticmethod
    def is_mfrsd1d2_content(text: str) -> bool:
        """Check if text IS a main document section header (exact match only)."""
        text_cleaned = text.strip()
        if len(text_cleaned) > 20:
            return False
        text_normalized = re.sub(r'\s+', '', text_cleaned)
        exact_patterns = ['主文', '事實', '理由', '事實及理由']
        return text_normalized in exact_patterns

    @staticmethod
    def is_date_content(text: str) -> bool:
        """Check if text IS a Chinese date header (not just contains dates)."""
        text_cleaned = text.strip()
        if len(text_cleaned) > 50:
            return False
        date_patterns = [
            r'^中\s*華\s*民\s*國.*\d+.*年.*\d+.*月.*\d+.*日$',
            r'^\d{3,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日$'
        ]
        return any(re.match(pattern, text_cleaned) for pattern in date_patterns)

    @staticmethod
    def is_between_d1_d2(chunk, sorted_chunks, idx: int) -> bool:
        """Check if a chunk is between D1 and D2 sections."""
        for i in range(max(0, idx - 5), min(len(sorted_chunks), idx + 6)):
            if i == idx:
                continue
            chunk_text = " ".join(sorted_chunks[i].content_lines)
            if re.search(r'D1|D2', chunk_text, re.IGNORECASE):
                return True
        return False

    def chunks_to_markdown(self, result) -> str:
        """Convert detection result chunks to Markdown format with new formatting rules."""
        if not result.line_based_chunks:
            raise ValueError("Detection result does not contain line based chunks.")

        output_lines: List[str] = [
            f"# {result.filename}",
            f"- learning_region: {result.learning_region}",
            ""
        ]
        numbering_state: Dict[int, int] = {}

        sorted_chunks = sorted(result.line_based_chunks, key=lambda item: item.start_line)
        date2_position = self.find_date2_position(sorted_chunks)

        for idx, chunk in enumerate(sorted_chunks):
            raw_lines = [line.rstrip("\n") for line in chunk.content_lines]
            is_in_footer = date2_position >= 0 and idx > date2_position

            if chunk.level == -3:
                header_footer_lines = []
                for line in raw_lines:
                    stripped = line.strip()
                    if stripped:
                        if (stripped.startswith('編號') and ('接辦案件' in stripped or '時間' in stripped or '對話時間' in stripped) or
                                stripped.startswith('刑法第') and '律師法第' in stripped):
                            continue
                        header_footer_lines.append(line)
                if not header_footer_lines:
                    continue
                output_lines.extend(header_footer_lines)
                output_lines.append("")
                continue

            content_lines = self.sanitize_lines(raw_lines)
            if not content_lines:
                continue

            joined_content = " ".join(content_lines)
            is_h2_candidate = False

            if self.is_mfrsd1d2_content(joined_content):
                is_h2_candidate = True
            elif self.is_date_content(joined_content):
                is_h2_candidate = True
            elif self.is_appendix_content(joined_content):
                is_h2_candidate = True

            if is_h2_candidate:
                if self.is_appendix_content(joined_content):
                    if len(content_lines) > 1:
                        first_line = content_lines[0] if content_lines else joined_content
                        output_lines.append(f"## {first_line}")
                        output_lines.append("")
                        for line in content_lines[1:]:
                            if line.strip():
                                output_lines.append(line)
                        output_lines.append("")
                    else:
                        output_lines.append(f"## {joined_content}")
                        output_lines.append("")
                else:
                    heading_text = joined_content
                    if not heading_text:
                        continue
                    output_lines.append(f"## {heading_text}")
                    output_lines.append("")
                numbering_state.clear()

            elif chunk.level == 0:
                paragraph = joined_content
                if paragraph:
                    output_lines.append(paragraph)
                    output_lines.append("")

            elif chunk.level >= 1:
                numbering_state.setdefault(chunk.level, 0)
                numbering_state[chunk.level] += 1
                for deeper_level in [lvl for lvl in numbering_state if lvl > chunk.level]:
                    numbering_state.pop(deeper_level, None)
                indent = "    " * (chunk.level - 1)
                numbered_text = f"{indent}{numbering_state[chunk.level]}. {joined_content}"
                output_lines.append(numbered_text)
                output_lines.append("")

            elif chunk.level == -1:
                if self.is_between_d1_d2(chunk, sorted_chunks, idx):
                    continue
                if self.is_appendix_content(joined_content):
                    line_by_line = [line for line in raw_lines if line.strip()]
                    if line_by_line:
                        output_lines.extend(line_by_line)
                        output_lines.append("")
                    continue
                prev_chunk = sorted_chunks[idx - 1] if idx > 0 else None
                next_chunk = sorted_chunks[idx + 1] if idx + 1 < len(sorted_chunks) else None
                adjacent_to_special = (
                    (prev_chunk and (prev_chunk.level == -3 or prev_chunk.chunk_type in ["header", "footer"]))
                    or (next_chunk and (next_chunk.level == -3 or next_chunk.chunk_type in ["header", "footer"]))
                )
                if adjacent_to_special:
                    line_by_line = [line for line in raw_lines if line.strip()]
                    if line_by_line:
                        output_lines.extend(line_by_line)
                        output_lines.append("")
                else:
                    paragraph = joined_content
                    if paragraph:
                        output_lines.append(paragraph)
                        output_lines.append("")

            elif chunk.level == -2:
                date_text = joined_content
                if date_text:
                    output_lines.append(f"*{date_text}*")
                    output_lines.append("")

        cleaned_output = [line.rstrip() for line in output_lines]
        return "\n".join(cleaned_output).strip() + "\n"

    def load_machine_detection_result(self, payload_path: Path):
        """Load detection result from machine export JSON."""
        data = json.loads(payload_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Empty result array in {payload_path}")
            data = data[0]

        learned_rules = [
            LevelingRule(
                symbol_category=rule.get("symbol_category", ""),
                assigned_level=rule.get("assigned_level", 0),
                confidence=rule.get("confidence", 0.0),
                learning_source=rule.get("learning_source", ""),
                occurrences=rule.get("occurrences", 0),
                examples=list(rule.get("examples", [])),
            )
            for rule in data.get("learned_rules", [])
        ]

        line_chunks = [
            LineBasedChunk(
                level=item.get("level", -1),
                start_line=item.get("start_line", 0),
                end_line=item.get("end_line", 0),
                chunk_type=item.get("chunk_type", "content"),
                content_lines=list(item.get("content_lines", [])),
                leveling_symbol=item.get("leveling_symbol"),
                chunk_id=item.get("chunk_id", ""),
            )
            for item in data.get("line_based_chunks", [])
        ]

        return AdaptiveDetectionResult(
            filename=data.get("filename", payload_path.name),
            file_structure=data.get("file_structure", {}),
            learning_region=data.get("learning_region", ""),
            learned_rules=learned_rules,
            full_detection_results=[],
            applied_hierarchy=data.get("applied_hierarchy", {}),
            processing_stats=data.get("processing_stats", {}),
            line_based_chunks=line_chunks,
        )

    def process_file(self, input_file: Path, machine_input: bool = False):
        """Process a single file and return the detection result."""
        if machine_input:
            return self.load_machine_detection_result(input_file)
        else:
            if self.detector is None:
                raise ValueError("Detector instance required when machine_input is False")
            return self.detector.process_single_file(input_file)

    def convert_to_markdown(
        self,
        input_file: Path,
        output_dir: Path,
        machine_input: bool = False,
    ) -> Optional[Path]:
        """Convert a single file to Markdown and save it."""
        result = self.process_file(input_file, machine_input)
        if not result:
            return None
        markdown = self.chunks_to_markdown(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / f"{input_file.stem}.md"
        target_path.write_text(markdown, encoding="utf-8")
        return target_path

    def convert_batch(
        self,
        input_files: List[Path],
        output_dir: Path,
        machine_input: bool = False,
    ) -> List[Path]:
        """Convert multiple files to Markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files: List[Path] = []
        for input_file in input_files:
            try:
                output_path = self.convert_to_markdown(input_file, output_dir, machine_input)
                if output_path is not None:
                    written_files.append(output_path)
            except Exception as exc:
                print(f"Failed to convert {input_file.name}: {exc}")
                continue
        return written_files
