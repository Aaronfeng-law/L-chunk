import pytest
import re
from src.lchunk.analyzers.splitter_refactor import *

# 測試文字正規化功能
def test_normalize_text():
    assert normalize_text("  中 華  \n 民 國  ") == "中華民國"
    assert normalize_text("  測試\r\n換行") == "測試換行"
    assert normalize_text("  ") == ""

# 測試 Pattern 定義是否包含必要的 Key
def test_find_section_pattern():
    patterns = find_section_pattern()
    assert 'date_pattern' in patterns
    assert 'main_text' in patterns
    # 測試 date_pattern 是否能正確匹配
    date_re = patterns['date_pattern']
    assert date_re.match("中華民國112年5月20日")
    assert date_re.match(normalize_text("中 華 民 國 112 年 5 月 20 日"))

def test_find_strict_date_indexes_strict_limit():
    patterns = find_section_pattern()
    date_pattern = patterns['date_pattern']
    
    test_lines = [
        "中華民國112年1月1日",                        # 成功案例：恰好一組
        "中華民國112年1月1日及中華民國113年1月1日",      # 失敗案例：兩組日期，應被排除
        "這行沒有日期",                                # 失敗案例：零組
        "中華民國112年1月1日 之後還有內容",             # 成功案例：內含一組 (findall 會抓到)
    ]
    
    results = find_strict_date_indexes(test_lines, date_pattern)
    
    # 預期結果：只有 index 0 和 index 3 符合（因為它們各只有一組日期）
    assert len(results) == 1
    assert results[0]['index'] == 0
    
    # 驗證內容
    assert results[0]['full_match'] == "中華民國112年1月1日"

def test_no_matches_found():
    """測試完全沒有日期時的回傳"""
    patterns = find_section_pattern()
    results = find_strict_date_indexes(["純文字", "12345"], patterns['date_pattern'])
    assert results == []
    
import pytest
import re

def test_integration_logic():
    patterns = find_section_pattern()
    date_pattern = patterns['date_pattern']
    
    test_lines = [
        "中　華　民　國　112　年　5　月　20　日", 
        "中華民國112年5月20日及113年1月1日",     # 這裡有兩組日期邏輯
        "  中華民國 112 年 5 月 20 日  "
    ]
    
    results = find_strict_date_indexes(test_lines, date_pattern)
    
    # 檢查結果是否排除掉了第 1 行
    matched_indices = [r['index'] for r in results]
    assert 1 not in matched_indices, f"索引 1 應該被排除，但結果卻包含了它：{results}"
    assert len(results) == 2

def test_normalize_text_extreme_spaces():
    # 測試是否能處理各種怪異空白
    input_str = "\u3000內容\u00A0帶有\t多種 空白\n\r"
    assert normalize_text(input_str) == "內容帶有多種空白"    

def test_strict_keyline_capture():
    lines = [
        "標題",                # 被忽略
        "主文",                # 關鍵行，記錄
        "中間無關文字",         # 被忽略
        "事實",                # 關鍵行，記錄
        "中間文字",            # 被忽略
        "理由",                # 關鍵行，記錄
        "中華民國112年1月1日",  # 日期1
        "中華民國112年12月31日",# 關鍵行(日期2)，記錄至footer
        "無關文字",            # 被忽略
        "附錄一",              # 關鍵行，記錄
        "無關文字",            # 被忽略
        "附表三"               # 關鍵行，記錄
    ]
    patterns = find_section_pattern()
    sections = classify_document_sections(lines, patterns)
    
    # 斷言只記錄了關鍵行
    assert len(sections['main_text']) == 1
    assert len(sections['facts']) == 1
    assert len(sections['reasons']) == 1
    assert len(sections['footer']) == 1
    assert len(sections['appendix']) == 2 # 附錄一 與 附表三
    
    # 確保內容正確
    assert sections['appendix'][0]['cleaned_line'] == "附錄一"