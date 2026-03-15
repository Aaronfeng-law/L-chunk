import pytest
import re
from src.lchunk.analyzers.splitter_refactor import (
    normalize_text,
    find_section_pattern,
    classify_document_sections,
    DocumentLine,
    SectionContent
)

# 測試文字正規化功能
def test_normalize_text():
    assert normalize_text("  中 華  \n 民 國  ") == "中華民國"
    assert normalize_text("  測試\r\n換行") == "測試換行"
    assert normalize_text("  ") == ""
    assert normalize_text(None) == ""

# 測試 Pattern 定義是否包含必要的 Key
def test_find_section_pattern():
    patterns = find_section_pattern()
    assert 'date_pattern' in patterns
    assert 'main_text' in patterns
    # 測試 date_pattern 是否能正確匹配
    date_re = patterns['date_pattern']
    assert date_re.match("中華民國112年5月20日")
    # 測試正規化後的匹配
    assert date_re.match("中華民國112年5月20日")

def test_normalize_text_extreme_spaces():
    # 測試是否能處理各種怪異空白
    input_str = "\u3000內容\u00A0帶有\t多種 空白\n\r"
    assert normalize_text(input_str) == "內容帶有多種空白"    

def test_classify_document_sections_flow():
    """測試文件分段邏輯與 Dataclass 結構"""
    lines = [
        "標題",                # header
        "主文",                # main_text trigger
        "被告應給付原告",       # main_text content
        "事實",                # facts trigger
        "中間文字",            # facts content
        "理由",                # reasons trigger
        "內容",                # reasons content
        "中華民國112年1月1日",  # 日期1 (reasons content)
        "中華民國112年12月31日",# 日期2 (trigger footer)
        "書記官",              # footer content
        "附錄一",              # appendix trigger
        "附表三"               # appendix content
    ]
    patterns = find_section_pattern()
    full_lines, sections = classify_document_sections(lines, patterns)
    
    # 驗證回傳型別
    assert isinstance(full_lines[0], DocumentLine)
    assert isinstance(sections['main_text'], SectionContent)
    
    # 驗證 Header
    assert len(sections['header'].lines) == 1
    assert sections['header'].lines[0].original_text == "標題"
    
    # 驗證 Main Text
    # 包含觸發行 "主文" 和內容 "被告應給付原告"
    assert len(sections['main_text'].lines) == 2
    assert sections['main_text'].lines[0].normalized_text == "主文"
    
    # 驗證 Facts
    assert len(sections['facts'].lines) == 2
    assert sections['facts'].lines[0].normalized_text == "事實"
    
    # 驗證 Reasons (應包含第一個日期)
    assert len(sections['reasons'].lines) == 3
    assert sections['reasons'].lines[0].normalized_text == "理由"
    assert sections['reasons'].lines[2].normalized_text == "中華民國112年1月1日"
    
    # 驗證 Footer (從第二個日期開始)
    # 第二個日期觸發切換到 footer，該行本身加入 footer
    assert len(sections['footer'].lines) == 2
    assert sections['footer'].lines[0].normalized_text == "中華民國112年12月31日"
    assert sections['footer'].lines[1].normalized_text == "書記官"
    
    # 驗證 Appendix
    assert len(sections['appendix'].lines) == 2
    assert sections['appendix'].lines[0].normalized_text == "附錄一"
    assert sections['appendix'].lines[1].normalized_text == "附表三"