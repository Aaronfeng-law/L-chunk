#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試特殊標記檢測的腳本
處理中文文字中的空格和Unicode字符（如\u3000）的問題
"""

import json
import re
import unicodedata


def load_test_file():
    """加載測試文件"""
    file_path = "/home/soogoino/Publics/Projects/L-chunk/data/samples/TPDM,111,金重訴,34,20250114,1.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"讀取文件失敗: {e}")
        return None


def normalize_text(text):
    """標準化文本，移除多餘空格和統一Unicode字符"""
    if not text:
        return text
    
    # 1. Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 將全角空格轉換為半角空格
    text = text.replace('\u3000', ' ')
    text = text.replace('\xa0', ' ')  # 不換行空格
    
    # 3. 移除行首行尾空格但保留行內空格
    text = text.strip()
    
    return text


def flexible_find_markers(text, markers):
    """
    靈活的標記查找函數，處理空格和Unicode變化
    """
    if not text or not markers:
        return []
    
    found_markers = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # 標準化行內容
        normalized_line = normalize_text(line)
        
        for marker in markers:
            # 標準化標記
            normalized_marker = normalize_text(marker)
            
            # 方法1: 直接匹配
            if normalized_marker in normalized_line:
                found_markers.append({
                    'marker': marker,
                    'line_number': i + 1,
                    'line_content': line.strip(),
                    'match_type': 'direct'
                })
                continue
            
            # 方法2: 移除所有空格後匹配
            line_no_spaces = re.sub(r'\s+', '', normalized_line)
            marker_no_spaces = re.sub(r'\s+', '', normalized_marker)
            
            if marker_no_spaces in line_no_spaces:
                found_markers.append({
                    'marker': marker,
                    'line_number': i + 1,
                    'line_content': line.strip(),
                    'match_type': 'no_spaces'
                })
                continue
            
            # 方法3: 使用正則表達式，允許空格變化
            # 將標記中的每個字符之間允許可選的空格
            pattern = ''
            for char in normalized_marker:
                if char.strip():  # 非空格字符
                    pattern += re.escape(char) + r'\s*'
                else:
                    pattern += r'\s+'
            
            pattern = pattern.rstrip(r'\s*')  # 移除末尾的 \s*
            
            if re.search(pattern, normalized_line):
                found_markers.append({
                    'marker': marker,
                    'line_number': i + 1,
                    'line_content': line.strip(),
                    'match_type': 'regex_flexible'
                })
    
    return found_markers


def test_original_markers():
    """測試原始標記檢測"""
    print("=== 測試原始標記檢測 ===")
    
    # 加載測試數據
    data = load_test_file()
    if not data:
        return
    
    # 從JFULL字段獲取全文
    full_text = data.get('JFULL', '')
    if not full_text:
        print("找不到JFULL字段")
        return
    
    print(f"文檔總長度: {len(full_text)} 字符")
    print(f"行數: {len(full_text.split('\n'))}")
    
    # 定義要查找的標記
    markers = ['主文', '事實', '理由']
    
    print(f"\n搜索標記: {markers}")
    
    # 使用靈活的標記查找
    found_markers = flexible_find_markers(full_text, markers)
    
    if found_markers:
        print(f"\n找到 {len(found_markers)} 個標記:")
        for marker_info in found_markers:
            print(f"- {marker_info['marker']} (第{marker_info['line_number']}行, {marker_info['match_type']})")
            print(f"  內容: {marker_info['line_content'][:100]}...")
            print()
    else:
        print("\n未找到任何標記")
        
        # 調試：顯示一些包含這些字符的行
        print("\n調試：查找包含相關字符的行...")
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            normalized_line = normalize_text(line)
            for marker in markers:
                # 檢查是否包含標記的任何字符
                for char in marker:
                    if char in normalized_line:
                        print(f"第{i+1}行包含'{char}': {line.strip()[:100]}...")
                        break


def test_context_around_markers():
    """測試標記周圍的上下文"""
    print("\n=== 測試標記周圍的上下文 ===")
    
    data = load_test_file()
    if not data:
        return
    
    full_text = data.get('JFULL', '')
    if not full_text:
        return
    
    markers = ['主文', '事實', '理由']
    found_markers = flexible_find_markers(full_text, markers)
    
    if found_markers:
        lines = full_text.split('\n')
        for marker_info in found_markers:
            line_num = marker_info['line_number'] - 1  # 轉換為0索引
            print(f"\n標記: {marker_info['marker']} (第{marker_info['line_number']}行)")
            
            # 顯示前後各3行的上下文
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 4)
            
            for i in range(start, end):
                prefix = ">>> " if i == line_num else "    "
                print(f"{prefix}{i+1:4d}: {lines[i].strip()}")


def show_unicode_analysis():
    """分析文本中的Unicode字符"""
    print("\n=== Unicode字符分析 ===")
    
    data = load_test_file()
    if not data:
        return
    
    full_text = data.get('JFULL', '')
    if not full_text:
        return
    
    # 查找特殊Unicode字符
    special_chars = set()
    for char in full_text:
        if ord(char) > 127:  # 非ASCII字符
            if char in [' ', '　', '\u3000', '\xa0']:  # 各種空格
                special_chars.add(f"'{char}' (U+{ord(char):04X}) - {unicodedata.name(char, 'UNKNOWN')}")
    
    if special_chars:
        print("發現的特殊空格字符:")
        for char_info in sorted(special_chars):
            print(f"  {char_info}")
    
    # 統計各種空格的數量
    space_count = full_text.count(' ')
    fullwidth_space_count = full_text.count('　')
    ideographic_space_count = full_text.count('\u3000')
    nbsp_count = full_text.count('\xa0')
    
    print(f"\n空格統計:")
    print(f"  普通空格 (U+0020): {space_count}")
    print(f"  全角空格 (U+3000): {fullwidth_space_count + ideographic_space_count}")
    print(f"  不換行空格 (U+00A0): {nbsp_count}")


if __name__ == "__main__":
    print("測試特殊標記檢測功能")
    print("=" * 50)
    
    test_original_markers()
    test_context_around_markers()
    show_unicode_analysis()
    
    print("\n測試完成！")