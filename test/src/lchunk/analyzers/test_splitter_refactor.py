import pytest
import re
from src.lchunk.analyzers.splitter_refactor import (
    normalize_text,
    find_section_pattern,
    classify_document_sections,
)
from src.lchunk.pipeline import DocumentLine, SectionContent

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
    assert date_re.match("中華民國112年5月20日")

def test_normalize_text_extreme_spaces():
    # 測試是否能處理各種怪異空白
    input_str = "\u3000內容\u00A0帶有\t多種 空白\n\r"
    assert normalize_text(input_str) == "內容帶有多種空白"    

def test_classify_document_sections_flow():
    """測試文件分段邏輯與 Dataclass 結構
    
    新狀態機行為：
      第1個日期 → date1（單行）
      date1 後的行 → sig（含法官簽名、書記官等）
      第2個日期 → date2（單行起，後續非 appendix 行仍在 date2）
      附[錄件圖表] → appendix
    """
    lines = [
        "標題",                # header
        "主文",                # main_text trigger
        "被告應給付原告",       # main_text content
        "事實",                # facts trigger
        "中間文字",            # facts content
        "理由",                # reasons trigger
        "內容",                # reasons content
        "中華民國112年1月1日",  # 日期1 → date1（單行）
        "刑事第九庭  審判長法官", # sig content
        "書記官  王小明",       # sig content
        "中華民國112年12月31日", # 日期2 → date2 起始行
        "附錄一",              # appendix trigger（從 date2 轉入 appendix）
        "附表三"               # appendix（仍是 appendix，不會觸發新分段，因為已在 appendix）
    ]
    patterns = find_section_pattern()
    full_lines, sections, key_lines = classify_document_sections(lines, patterns)

    # 驗證回傳型別
    assert isinstance(full_lines[0], DocumentLine)
    assert isinstance(sections['main_text'], SectionContent)

    # ── header ───────────────────────────────────────────────────────
    assert len(sections['header'].lines) == 1
    assert sections['header'].lines[0].original_text == "標題"

    # ── main_text：觸發行 "主文" + 內容 ─────────────────────────────
    assert len(sections['main_text'].lines) == 2
    assert sections['main_text'].lines[0].normalized_text == "主文"

    # ── facts：觸發行 "事實" + 內容 ──────────────────────────────────
    assert len(sections['facts'].lines) == 2
    assert sections['facts'].lines[0].normalized_text == "事實"

    # ── reasons：觸發行 "理由" + 內容（不含日期行）─────────────────
    assert len(sections['reasons'].lines) == 2
    assert sections['reasons'].lines[0].normalized_text == "理由"
    assert sections['reasons'].lines[1].normalized_text == "內容"

    # ── date1：第一個日期行（單行，獨立 section）────────────────────
    assert len(sections['date1'].lines) == 1
    assert sections['date1'].lines[0].normalized_text == "中華民國112年1月1日"

    # ── sig：date1 之後到 date2 之前（法官/書記官簽名區）───────────
    assert len(sections['sig'].lines) == 2
    assert sections['sig'].lines[0].normalized_text == "刑事第九庭審判長法官"
    assert sections['sig'].lines[1].normalized_text == "書記官王小明"

    # ── date2：第二個日期行（正本送達日期）──────────────────────────
    # 注意：date2 只有 1 行，附錄觸發行轉入 appendix
    assert len(sections['date2'].lines) == 1
    assert sections['date2'].lines[0].normalized_text == "中華民國112年12月31日"

    # ── appendix ─────────────────────────────────────────────────────
    assert len(sections['appendix'].lines) == 2
    assert sections['appendix'].lines[0].normalized_text == "附錄一"
    assert sections['appendix'].lines[1].normalized_text == "附表三"

    # ── footer 應不存在（已廢棄）─────────────────────────────────────
    assert 'footer' not in sections


def test_date1_is_single_line():
    """date1 section 只含一行（日期行本身），後續進 sig。"""
    lines = [
        "主文",
        "判決主文內容",
        "中華民國113年6月1日",  # date1
        "法官  張三",           # sig
        "書記官  李四",          # sig
        "中華民國113年6月2日",  # date2
    ]
    patterns = find_section_pattern()
    _, sections, _ = classify_document_sections(lines, patterns)

    assert len(sections['date1'].lines) == 1
    assert "113年6月1日" in sections['date1'].lines[0].normalized_text

    assert len(sections['sig'].lines) == 2

    assert len(sections['date2'].lines) == 1
    assert "113年6月2日" in sections['date2'].lines[0].normalized_text


def test_appendix_split_between_date2_and_appendix():
    """appendix 緊接在 date2 之後，可正確切換。"""
    lines = [
        "主文",
        "中華民國113年1月1日",   # date1
        "書記官",               # sig
        "中華民國113年1月2日",   # date2
        "附錄本案論罪條文：",    # appendix 觸發
        "刑法第268條",          # appendix 內容
        "附件：",               # 新的附錄物件觸發
        "卷宗標目",             # appendix 內容
    ]
    patterns = find_section_pattern()
    _, sections, _ = classify_document_sections(lines, patterns)

    assert len(sections['date2'].lines) == 1
    assert len(sections['appendix'].lines) == 4  # 觸發行+內容共4行