#!/usr/bin/env python3
"""Markdown converter for adaptive detection results."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..detectors.adaptive_hybrid import (
    AdaptiveDetectionResult,
    AdaptiveHybridDetector,
    LevelingRule,
    LineBasedChunk,
)


class MarkdownConverter:
    """Convert adaptive detection results to Markdown format."""

    def __init__(self, detector: AdaptiveHybridDetector | None = None, model_path: Path | str | None = None):
        """Initialize the converter with an optional detector instance or model path."""
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
        
        # Must be a reasonably short line that looks like an appendix header
        if len(text_cleaned) > 100:  # Appendix headers are typically short
            return False
            
        # Check for appendix patterns at the start of the line
        appendix_patterns = [r'^附錄', r'^附件', r'^附圖', r'^附表']
        text_normalized = re.sub(r'\s+', '', text_cleaned)
        appendix_patterns_normalized = [r'^附錄', r'^附件', r'^附圖', r'^附表']
        
        return (any(re.match(pattern, text_cleaned) for pattern in appendix_patterns) or
                any(re.match(pattern, text_normalized) for pattern in appendix_patterns_normalized))
    
    @staticmethod
    def is_standalone_appendix_header(text: str, is_in_footer: bool = False) -> bool:
        """Check if text is a standalone appendix header in footer section."""
        if not is_in_footer:
            return False
        standalone_patterns = [r'^附錄.*', r'^附件.*', r'^附圖.*', r'^附表[一二三四五六七八九十\d]*$']
        return any(re.match(pattern, text.strip()) for pattern in standalone_patterns)

    @staticmethod
    def find_date2_position(sorted_chunks: List[LineBasedChunk]) -> int:
        """Find the position of Date2 section in the chunks."""
        for i, chunk in enumerate(sorted_chunks):
            content_text = " ".join(chunk.content_lines)
            # Look for Date2 patterns - adjust these patterns based on your data
            if re.search(r'中\s*華\s*民\s*國.*年.*月.*日', content_text) or chunk.level == -2:
                return i
        return -1  # Date2 not found

    @staticmethod
    def is_mfrsd1d2_content(text: str) -> bool:
        """Check if text IS a main document section header (exact match only)."""
        text_cleaned = text.strip()
        
        # Must be a short line that looks like a section header
        if len(text_cleaned) > 20:  # Section headers are typically very short
            return False
            
        # Remove all spaces and check for exact match
        text_normalized = re.sub(r'\s+', '', text_cleaned)
        
        exact_patterns = [
            '主文',
            '事實', 
            '理由', 
            '事實及理由'
        ]
        return text_normalized in exact_patterns

    @staticmethod
    def is_date_content(text: str) -> bool:
        """Check if text IS a Chinese date header (not just contains dates)."""
        text_cleaned = text.strip()
        
        # Must be a short line that looks like a date header
        if len(text_cleaned) > 50:  # Date headers are typically short
            return False
            
        # Pattern for Chinese formal date - must be the main content of the line
        date_patterns = [
            r'^中\s*華\s*民\s*國.*\d+.*年.*\d+.*月.*\d+.*日$',
            r'^\d{3,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日$'
        ]
        
        # Check if text matches date pattern exactly (anchored to start and end)
        return any(re.match(pattern, text_cleaned) for pattern in date_patterns)

    @staticmethod
    def is_between_d1_d2(chunk: LineBasedChunk, sorted_chunks: List[LineBasedChunk], idx: int) -> bool:
        """Check if a chunk is between D1 and D2 sections."""
        # Look for D1 and D2 patterns in surrounding chunks
        for i in range(max(0, idx - 5), min(len(sorted_chunks), idx + 6)):
            if i == idx:
                continue
            chunk_text = " ".join(sorted_chunks[i].content_lines)
            if re.search(r'D1|D2', chunk_text, re.IGNORECASE):
                return True
        return False

    def chunks_to_markdown(self, result: AdaptiveDetectionResult) -> str:
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
        
        # Find Date2 position to determine footer section
        date2_position = self.find_date2_position(sorted_chunks)

        for idx, chunk in enumerate(sorted_chunks):
            raw_lines = [line.rstrip("\n") for line in chunk.content_lines]
            
            # Determine if we're in the footer section (after Date2)
            is_in_footer = date2_position >= 0 and idx > date2_position

            # Handle header/footer (level -3)
            if chunk.level == -3:
                header_footer_lines = []
                for line in raw_lines:
                    stripped = line.strip()
                    if stripped:
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

            # Only very specific patterns should be H2 headers
            is_h2_candidate = False
            
            # Check for main document sections (MFRSD1D2)
            if self.is_mfrsd1d2_content(joined_content):
                is_h2_candidate = True
                
            # Check for date lines (level 0 or -2 in adaptive_hybrid)
            elif self.is_date_content(joined_content):
                is_h2_candidate = True
                
            # Check for appendix content anywhere (not limited to footer)
            elif self.is_appendix_content(joined_content):
                is_h2_candidate = True
            
            if is_h2_candidate:
                # Check if it's appendix content for special handling
                if self.is_appendix_content(joined_content):
                    # For appendix, use H2 but maintain line-by-line structure if multiple lines
                    if len(content_lines) > 1:
                        first_line = content_lines[0] if content_lines else joined_content
                        output_lines.append(f"## {first_line}")
                        output_lines.append("")
                        
                        # Add remaining lines as-is (line-by-line structure)
                        for line in content_lines[1:]:
                            if line.strip():
                                output_lines.append(line)
                        output_lines.append("")
                    else:
                        # Single line appendix
                        output_lines.append(f"## {joined_content}")
                        output_lines.append("")
                else:
                    # Regular H2 heading for MFRSD1D2 and dates
                    heading_text = joined_content
                    if not heading_text:
                        continue
                    output_lines.append(f"## {heading_text}")
                    output_lines.append("")
                
                # Reset numbering for new sections
                numbering_state.clear()
            
            # Handle level 0 content that doesn't match H2 patterns
            elif chunk.level == 0:
                # Regular content, not H2 - treat as paragraph
                paragraph = joined_content
                if paragraph:
                    output_lines.append(paragraph)
                    output_lines.append("")

            # Handle numbered levels (1, 2, 3, 4, 5, 6, 7, ...)
            elif chunk.level >= 1:
                # Update numbering state
                numbering_state.setdefault(chunk.level, 0)
                numbering_state[chunk.level] += 1

                # Reset deeper level counters
                for deeper_level in [lvl for lvl in numbering_state if lvl > chunk.level]:
                    numbering_state.pop(deeper_level, None)

                # Create indentation and numbering
                indent = "    " * (chunk.level - 1)
                numbered_text = f"{indent}{numbering_state[chunk.level]}. {joined_content}"
                output_lines.append(numbered_text)
                output_lines.append("")

            # Handle level -1 (concatenate except header, footer, text between D1-D2, and appendix)
            elif chunk.level == -1:
                # Skip if it's between D1 and D2
                if self.is_between_d1_d2(chunk, sorted_chunks, idx):
                    continue

                # Check if this is appendix content in footer - preserve line-by-line structure
                if self.is_appendix_content(joined_content):
                    line_by_line = [line for line in raw_lines if line.strip()]
                    if line_by_line:
                        output_lines.extend(line_by_line)
                        output_lines.append("")
                    continue

                # Check if adjacent to header/footer
                prev_chunk = sorted_chunks[idx - 1] if idx > 0 else None
                next_chunk = sorted_chunks[idx + 1] if idx + 1 < len(sorted_chunks) else None

                adjacent_to_special = (
                    (prev_chunk and (prev_chunk.level == -3 or prev_chunk.chunk_type in ["header", "footer"]))
                    or (next_chunk and (next_chunk.level == -3 or next_chunk.chunk_type in ["header", "footer"]))
                )

                if adjacent_to_special:
                    # Don't concatenate, keep line-by-line
                    line_by_line = [line for line in raw_lines if line.strip()]
                    if line_by_line:
                        output_lines.extend(line_by_line)
                        output_lines.append("")
                else:
                    # Concatenate as paragraph
                    paragraph = joined_content
                    if paragraph:
                        output_lines.append(paragraph)
                        output_lines.append("")

            # Handle level -2 (dates, keep as italics)
            elif chunk.level == -2:
                date_text = joined_content
                if date_text:
                    output_lines.append(f"*{date_text}*")
                    output_lines.append("")

        # Clean up output
        cleaned_output = [line.rstrip() for line in output_lines]
        return "\n".join(cleaned_output).strip() + "\n"

    def load_machine_detection_result(self, payload_path: Path) -> AdaptiveDetectionResult:
        """Load detection result from machine export JSON."""
        data = json.loads(payload_path.read_text(encoding="utf-8"))
        
        # Handle both single result and array formats
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

        # Try to load from rag_chunks first (new format), fallback to line_based_chunks (old format)
        chunks_data = data.get("rag_chunks", data.get("line_based_chunks", []))
        
        line_chunks = []
        for item in chunks_data:
            # If content_lines is empty but content field exists, split content into lines
            content_lines = list(item.get("content_lines", []))
            if not content_lines and "content" in item:
                content_lines = item["content"].split("\n")
            
            # For hierarchical sections (level >= 1), prepend the title (leveling symbol line)
            # because the title is the leveling symbol line itself
            if item.get("level", -1) >= 1 and item.get("chunk_type") == "hierarchical_section":
                title = item.get("title", "")
                if title:
                    # Prepend the title as the first line
                    content_lines = [title] + content_lines
            
            line_chunks.append(
                LineBasedChunk(
                    level=item.get("level", -1),
                    start_line=item.get("start_line", 0),
                    end_line=item.get("end_line", 0),
                    chunk_type=item.get("chunk_type", "content"),
                    content_lines=content_lines,
                    leveling_symbol=item.get("leveling_symbol"),
                    chunk_id=item.get("chunk_id", ""),
                )
            )

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

    def process_file(
        self,
        input_file: Path,
        machine_input: bool = False,
    ) -> AdaptiveDetectionResult | None:
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
    ) -> Path | None:
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
