#!/usr/bin/env python3
"""
Judgment Document Splitter (Phase 1 of Pipeline)
負責：讀取 JSON 判決文件 → 正規化文字 → 段落分割 → 提取 metadata → 產出 JudgmentArtifact

此模組是整個管線的第一站，產出的 JudgmentArtifact 作為唯一 DTO 傳遞給後續階段。
"""

import json
import sys
import re
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple

# ── 直接執行時自動設定 sys.path（uv run src/... 或 python src/...）──────────
# parents[0]=analyzers, [1]=lchunk, [2]=src, [3]=專案根目錄
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 從 pipeline.py 匯入統一的資料結構
from src.lchunk.pipeline import DocumentLine, SectionContent, JudgmentArtifact

def normalize_text(text):
    if not text:
        return ""
    # 1. 將常見的換行符與 Tab 替換為空字串
    # 2. 使用 \s 匹配所有 Unicode 空白字元（包含全形空白、不換行空白等）
    cleaned = re.sub(r'[\r\n\t\s\u3000\u00A0]+', '', text)
    return cleaned.strip()

def find_section_pattern():
    """Define the patterns found in the specified lines"""
    patterns = {
        'main_text': "主文",
        'facts': "事實",
        "reasons": "理由",
        'facts_and_reasons_pattern': re.compile(r'^\s*事實\s*[及與和]\s*理由\s*$'),
        'date_pattern': re.compile(r'^中華民國(\d+)年(\d+)月(\d+)日$'),
        'appendix': re.compile(r'^\s*附[錄件圖表]')
    }
    
    return patterns

def classify_document_sections(lines: List[str], patterns: Dict[str, Any]) -> Tuple[List[DocumentLine], Dict[str, SectionContent], List[DocumentLine]]:
    """
    Classifies lines into sections based on state machine logic.
    Returns:
        full_lines: List of all DocumentLine objects
        sections: Dictionary mapping section names to SectionContent
        key_lines: List of key lines that triggered section changes
    """
    full_lines = []
    key_lines = []
    # footer 已廢棄；date1 = 判決日期行，date2 = 正本送達日期行
    sections = {
        k: SectionContent(name=k)
        for k in ['header', 'main_text', 'facts_and_reasons', 'facts', 'reasons',
                  'date1', 'sig', 'date2', 'appendix']
    }

    current_state = 'header'
    date_found_count = 0

    for i, line in enumerate(lines):
        cleaned_line = normalize_text(line)
        doc_line = DocumentLine(index=i, original_text=line, normalized_text=cleaned_line)
        full_lines.append(doc_line)

        # ── next_state 初始化 ───────────────────────────────────────────────
        # date1 是單行 section：日期行本身進 date1，下一行自動進 sig
        if current_state == 'date1':
            next_state = 'sig'
        else:
            next_state = current_state


        if current_state == 'header':
            if patterns['main_text'] == cleaned_line:
                key_lines.append(doc_line)
                next_state = 'main_text'

        elif current_state == 'main_text':
            if patterns['facts_and_reasons_pattern'].match(cleaned_line):
                key_lines.append(doc_line)
                next_state = 'facts_and_reasons'
            elif patterns['facts'] == cleaned_line:
                key_lines.append(doc_line)
                next_state = 'facts'
            elif patterns['reasons'] == cleaned_line:
                key_lines.append(doc_line)
                next_state = 'reasons'

        elif current_state == 'facts':
            if patterns['reasons'] == cleaned_line:
                key_lines.append(doc_line)
                next_state = 'reasons'

        # ── 日期偵測：date1 → sig → date2 ───────────────────────────
        # 只在尚未進入 date2 / appendix 前偵測
        if next_state not in ('date2', 'appendix'):
            if patterns['date_pattern'].search(cleaned_line):
                key_lines.append(doc_line)
                date_found_count += 1
                if date_found_count == 1:
                    # 第一個日期行 → date1（單行），下一行進入 sig
                    next_state = 'date1'
                else:
                    # 第二個（含）以後的日期行 → date2
                    next_state = 'date2'


        # ── Appendix 偵測：在 sig / date2 / appendix 之後偵測 ────────
        if next_state in ('sig', 'date2', 'appendix'):
            if patterns['appendix'].match(cleaned_line):
                key_lines.append(doc_line)
                next_state = 'appendix'

        current_state = next_state
        sections[current_state].lines.append(doc_line)

    return full_lines, sections, key_lines
    

def split_judgment_document(file_path: Path) -> Optional[JudgmentArtifact]:
    """
    Splits the judgment document into sections based on identified sections.
    Returns a strongly-typed JudgmentArtifact.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'JFULL' not in data:
            return None
        
        # Using splitlines() handles various line endings (\n, \r\n) automatically
        lines = data['JFULL'].splitlines()
        patterns = find_section_pattern()
        
        full_lines, sections, key_lines= classify_document_sections(lines, patterns)
        
        #TODO: 使用額外的 metadata extractor 替代 直接從 JSON 提取 metadata 的方式，以減少記憶體使用
        # Extract metadata from JSON (excluding JFULL to save memory if needed)
        metadata = {k: v for k, v in data.items() if k != 'JFULL'}
        court_code = metadata.get('JID', '')[0:4] if 'JID' in metadata else ''
        
        # Optionally Map Court Names
        try:
            with open("data/court_codes/court_mapping_grouped.json", "r", encoding="utf-8") as f:
                court_data = json.load(f)
            for base_court_name, court_info in court_data.get("courts", {}).items():
                case_types = court_info.get("case_types", {})
                for case_type, type_info in case_types.items():
                    if type_info.get("sub_court_code") == court_code:
                        metadata['court_full_name'] = type_info.get("full_name")
                        metadata['base_court_name'] = base_court_name
                        metadata['case_type'] = case_type
                        break
                if 'court_full_name' in metadata:
                    break
        except FileNotFoundError:
            pass  # Ignore if court_codes mapping doesn't exist
        
        if court_code:
            print(f"Extracted metadata for court code {court_code}: {metadata.get('court_full_name', 'Unknown Court')}")
        
        return JudgmentArtifact(
            file_path=str(file_path),
            full_lines=full_lines,
            sections=sections,
            metadata=metadata,
            key_lines=key_lines
        )
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def process_single_file(file_path: Path) -> Optional[JudgmentArtifact]:
    """Wrapper function to process a single file and return the artifact"""
    return split_judgment_document(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split judgment document")
    parser.add_argument("input_file", type=Path, help="Path to input JSON file")
    parser.add_argument("--save", action="store_true", help="Save the result to a JSON file for debug, default save to output/debug/splitter_refactor_debug.json")
    
    args = parser.parse_args()
    
    artifact = process_single_file(args.input_file)
    
    try:
        if artifact:
            print(f"Processed {args.input_file} successfully.")
            # For demonstration, we can print out the sections and their line counts
            for section_name, content in artifact.sections.items():
                print(f"Section '{section_name}': {len(content.lines)} lines")
            for key_line in artifact.key_lines:
                print(f"Key Line (index {key_line.index}): {key_line.original_text}")
            
            if args.save:
                output_path = "output/debug/splitter_refactor_debug.json" # Default path for debug output
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(artifact), f, ensure_ascii=False, indent=2)
                print(f"Saved debug output to {output_path}")
        else:
            print(f"No valid JFULL content found in {args.input_file}.")
    except Exception as e:
        print(f"Error occurred while processing {args.input_file}: {e}")
