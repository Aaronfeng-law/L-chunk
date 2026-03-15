#!/usr/bin/env python3
"""
Judgment Document Splitter
Splits judgment documents based on structural patterns found in lines 40, 58, 100, 1504, 1514
"""

import json
import sys
import re
from pathlib import Path
from dataclasses import dataclasses, field
from typing import List, Dict, Any, Optional

@dataclasses(slots=True)




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


#回傳符合日期 pattern 的行號及日期內容，必須保持每一行（包含空白），避免Index錯誤，反正在markdown裡面也不會有空白行，這樣就不會有問題了

def find_strict_date_indexes(lines, date_pattern=None):
    """回傳所有符合嚴格日期 pattern 的行號及日期內容"""
    results = []
    for idx, line in enumerate(lines):
        normalized = normalize_text(line)
        match = date_pattern.findall(normalized)
        if len(match) == 1:
            year, month, day = match[0]
            results.append({
                'index': idx,
                'year': year,
                'month': month,
                'day': day,
                'full_match': f"中華民國{year}年{month}月{day}日",
                'original_line': line,
                'cleaned_line': normalized
            })
    return results

def classify_document_sections(lines, patterns):
    sections = {k: [] for k in ['header', 'main_text', 'facts_and_reasons', 'facts', 'reasons', 'footer', 'appendix']}
    
    current_state = 'header'
    date_found_count = 0

    for i, line in enumerate(lines):
        cleaned_line = normalize_text(line)
        
        # 狀態機與順序過濾邏輯
        if current_state == 'header':
            if patterns['main_text'] == cleaned_line:
                sections['main_text'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
                current_state = 'main_text'
                
        elif current_state == 'main_text':
            if patterns['facts_and_reasons_pattern'].match(cleaned_line):
                sections['facts_and_reasons'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
                current_state = 'facts_and_reasons'
            elif patterns['facts'] == cleaned_line:
                sections['facts'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
                current_state = 'facts'
                
        elif current_state == 'facts':
            if patterns['reasons'] == cleaned_line:
                sections['reasons'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
                current_state = 'reasons'
        
        # 日期檢測：監測日期觸發 footer
        # 只要仍在非 footer 狀態，且符合日期規則，即計數
        if current_state not in ['footer', 'appendix']:
            if patterns['date_pattern'].search(cleaned_line):
                date_found_count += 1
                if date_found_count >= 2:
                    sections['footer'].append({'index': i, 'original_line': line,"cleaned_line": cleaned_line})
                    current_state = 'footer'
        
        # Appendix 檢測：持續在 footer 之後檢測所有符合 appendix 模式的關鍵行
        elif current_state == 'footer':
            if patterns['appendix'].match(cleaned_line):
                sections['appendix'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
                current_state = 'appendix'
        
        elif current_state == 'appendix':
            if patterns['appendix'].match(cleaned_line):
                sections['appendix'].append({'index': i, 'original_line': line, "cleaned_line": cleaned_line})
            
    return sections
    

def split_judgment_document(jfull_content):
    """Splits the judgment document into sections based on identified sections"""
    