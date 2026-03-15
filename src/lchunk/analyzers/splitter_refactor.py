#!/usr/bin/env python3
"""
Judgment Document Splitter
Splits judgment documents based on structural patterns found in lines 40, 58, 100, 1504, 1514
"""

import json
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass(slots=True)
class DocumentLine:
    index: int
    original_text: str
    normalized_text: str
    tags: List[str] = field(default_factory=list)
    
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
    sections = {k: SectionContent(name=k) for k in ['header', 'main_text', 'facts_and_reasons', 'facts', 'reasons', 'sig', 'footer', 'appendix']}
    
    current_state = 'header'
    date_found_count = 0

    for i, line in enumerate(lines):
        cleaned_line = normalize_text(line)
        doc_line = DocumentLine(index=i, original_text=line, normalized_text=cleaned_line)
        full_lines.append(doc_line)
        
        next_state = current_state
        
        # 狀態機與順序過濾邏輯
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
        
        # 日期檢測：監測日期觸發 sig 與 footer
        # 只要仍在非 footer 狀態，且符合日期規則，即計數
        if next_state not in ['footer', 'appendix']:
            if patterns['date_pattern'].search(cleaned_line):
                key_lines.append(doc_line)
                date_found_count += 1
                if date_found_count == 1:
                    next_state = 'sig'
                elif date_found_count >= 2:
                    next_state = 'footer'
        
        # Appendix 檢測：持續在 sig/footer 之後檢測所有符合 appendix 模式的關鍵行
        # Allow transition from sig/footer to appendix, or within appendix (implicit)
        if next_state in ['sig', 'footer', 'appendix']:
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
        
        # Extract metadata from JSON (excluding JFULL to save memory if needed)
        metadata = {k: v for k, v in data.items() if k != 'JFULL'}
        
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
    if len(sys.argv) != 2:
        print("Usage: python splitter_refactor.py <path_to_json_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    artifact = process_single_file(file_path)
    
    try:
        if artifact:
            print(f"Processed {file_path} successfully.")
            # For demonstration, we can print out the sections and their line counts
            for section_name, content in artifact.sections.items():
                print(f"Section '{section_name}': {len(content.lines)} lines")
            for key_line in artifact.key_lines:
                print(f"Key Line (index {key_line.index}): {key_line.original_text}")
        else:
            print(f"No valid JFULL content found in {file_path}.")
    except Exception as e:
        print(f"Error occurred while processing {file_path}: {e}")
