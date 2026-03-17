#!/usr/bin/env python3
"""
Comprehensive Analysis Script for L-chunk Project
生成完整的判決書分析統計報告
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import re
import time
from datetime import datetime

# Import from judgment_splitter
sys.path.append('.')
from .splitter import process_single_file, find_section_patterns, extract_dates_from_text

def analyze_filtered_dataset():
    """分析完整的 filtered 數據集"""
    filtered_dir = Path("data/processed/filtered")
    
    if not filtered_dir.exists():
        print(f"❌ Filtered directory {filtered_dir} not found")
        return None
    
    json_files = list(filtered_dir.glob("*.json"))
    print(f"📊 Found {len(json_files)} JSON files in filtered dataset")
    
    # 統計數據
    stats = {
        'total_files': len(json_files),
        'successful_files': 0,
        'failed_files': 0,
        'section_stats': defaultdict(list),
        'case_types': Counter(),
        'year_distribution': Counter(),
        'court_types': Counter(),
        'processing_errors': [],
        'section_presence': Counter(),
        'empty_sections': Counter(),
        'file_sizes': [],
        'date_extraction_stats': {
            'files_with_dates': 0,
            'total_dates_found': 0,
            'date_patterns': Counter()
        },
        # 新增：符號化文件分類統計
        'file_categories': {
            # M=主文, F=事實, R=理由, D1=第一個日期, D2=第二個日期, N=無內容
            # 例如: MFR = 有主文+事實+理由, MF = 有主文+事實, R = 只有理由, N = 無內容
        },
        'symbol_categories': {},            # 動態生成的符號分類
        'detailed_stats': {
            'has_header': 0,
            'has_main_text': 0,
            'has_facts': 0,
            'has_reasons': 0,
            'has_facts_and_reasons': 0,
            'has_footer': 0
        }
    }
    
    print("🔄 Processing files...")
    start_time = time.time()
    
    for i, json_file in enumerate(json_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%)")
        
        try:
            # 分析檔案名稱模式 (TPDM,年度,案件類型,編號,日期,版本.json)
            filename_parts = json_file.stem.split(',')
            if len(filename_parts) >= 6:
                court = filename_parts[0]
                year = filename_parts[1]
                case_type = filename_parts[2]
                case_number = filename_parts[3]
                date_str = filename_parts[4]
                version = filename_parts[5]
                
                stats['year_distribution'][year] += 1
                stats['case_types'][case_type] += 1
                stats['court_types'][court] += 1
            
            # 處理檔案內容
            success, result = process_single_file(json_file)
            
            if success:
                # 讀取原始 JSON 數據以獲取全文內容
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats['successful_files'] += 1
                
                # 記錄各章節統計
                for section_name, content in result['sections'].items():
                    line_count = len(content) if content else 0
                    stats['section_stats'][section_name].append(line_count)
                    
                    if line_count > 0:
                        stats['section_presence'][section_name] += 1
                    else:
                        stats['empty_sections'][section_name] += 1
                
                # 新增：詳細的章節存在統計
                sections = result['sections']
                has_header = bool(sections.get('header', []))
                has_main_text = bool(sections.get('main_text', []))
                has_facts = bool(sections.get('facts', []))
                has_reasons = bool(sections.get('reasons', []))
                has_facts_and_reasons = bool(sections.get('facts_and_reasons', []))
                has_footer = bool(sections.get('footer', []))
                
                # 更新詳細統計
                if has_header: stats['detailed_stats']['has_header'] += 1
                if has_main_text: stats['detailed_stats']['has_main_text'] += 1
                if has_facts: stats['detailed_stats']['has_facts'] += 1
                if has_reasons: stats['detailed_stats']['has_reasons'] += 1
                if has_facts_and_reasons: stats['detailed_stats']['has_facts_and_reasons'] += 1
                if has_footer: stats['detailed_stats']['has_footer'] += 1
                
                # 文件分類邏輯 - 每個文件只會被分到一個類別
                file_info = {
                    'filename': json_file.name,
                    'has_header': has_header,
                    'has_main_text': has_main_text,
                    'has_facts': has_facts,
                    'has_reasons': has_reasons,
                    'has_facts_and_reasons': has_facts_and_reasons,
                    'has_footer': has_footer,
                    'facts_lines': len(sections.get('facts', [])),
                    'reasons_lines': len(sections.get('reasons', [])),
                    'combined_lines': len(sections.get('facts_and_reasons', [])),
                    'main_text_lines': len(sections.get('main_text', []))
                }
                
                # 日期提取統計 - 使用全文檢測更可靠
                content_text = data['JFULL']
                dates_found = extract_dates_from_text(content_text)
                
                # 生成符號化分類 - 修正版
                symbol = ""
                if has_main_text:
                    symbol += "M"
                if has_facts_and_reasons:
                    symbol += "S"  # S = Simplified (事實及理由合併)
                elif has_facts:
                    symbol += "F"
                if has_reasons and not has_facts_and_reasons:
                    symbol += "R"  # 只有當沒有合併時才加R
                
                # 日期檢測 - 簡化為是否有日期
                if dates_found:
                    if len(dates_found) >= 2:
                        symbol += "D2"  # 有多個日期
                    else:
                        symbol += "D1"  # 有一個日期
                
                # 如果沒有任何內容
                if not symbol or symbol == "":
                    symbol = "N"
                
                # 將文件歸類到對應的符號類別
                if symbol not in stats['symbol_categories']:
                    stats['symbol_categories'][symbol] = []
                
                file_info = {
                    'filename': json_file.name,
                    'symbol': symbol,
                    'has_header': has_header,
                    'has_main_text': has_main_text,
                    'has_facts': has_facts,
                    'has_reasons': has_reasons,
                    'has_facts_and_reasons': has_facts_and_reasons,
                    'has_footer': has_footer,
                    'facts_lines': len(sections.get('facts', [])),
                    'reasons_lines': len(sections.get('reasons', [])),
                    'combined_lines': len(sections.get('facts_and_reasons', [])),
                    'main_text_lines': len(sections.get('main_text', [])),
                    'dates_count': len(dates_found),
                    'header_lines': len(sections.get('header', [])),
                    'footer_lines': len(sections.get('footer', []))
                }
                
                stats['symbol_categories'][symbol].append(file_info)
                
                # 保持舊的分類邏輯用於向後兼容
                stats['file_categories'] = stats.get('file_categories', {})
                if symbol not in stats['file_categories']:
                    stats['file_categories'][symbol] = []
                stats['file_categories'][symbol].append(file_info)
                
                if dates_found:
                    stats['date_extraction_stats']['files_with_dates'] += 1
                    stats['date_extraction_stats']['total_dates_found'] += len(dates_found)
                    for date in dates_found:
                        if isinstance(date, dict) and 'year' in date and 'month' in date:
                            stats['date_extraction_stats']['date_patterns'][f"{date['year']}年{date['month']}月"] += 1
                        elif isinstance(date, tuple) and len(date) >= 2:
                            stats['date_extraction_stats']['date_patterns'][f"{date[0]}年{date[1]}月"] += 1
                
                # 檔案大小統計
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    stats['file_sizes'].append(len(content))
                    
            else:
                stats['failed_files'] += 1
                error_info = {
                    'filename': json_file.name,
                    'symbol': 'ERROR',
                    'error': str(result)
                }
                stats['processing_errors'].append(error_info)
                # 錯誤文件也加入符號分類
                if 'ERROR' not in stats['symbol_categories']:
                    stats['symbol_categories']['ERROR'] = []
                stats['symbol_categories']['ERROR'].append(error_info)
                
        except Exception as e:
            stats['failed_files'] += 1
            error_info = {
                'filename': json_file.name,
                'symbol': 'ERROR',
                'error': f"Exception: {str(e)}"
            }
            stats['processing_errors'].append(error_info)
            # 錯誤文件也加入符號分類
            if 'ERROR' not in stats['symbol_categories']:
                stats['symbol_categories']['ERROR'] = []
            stats['symbol_categories']['ERROR'].append(error_info)
    
    processing_time = time.time() - start_time
    stats['processing_time'] = processing_time
    
    return stats

def generate_comprehensive_report(stats):
    """生成綜合統計報告"""
    if not stats:
        return
    
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# L-chunk 判決書分析綜合報告
生成時間: {report_time}
處理時間: {stats['processing_time']:.2f} 秒

## 📊 整體統計

### 檔案處理統計
- 總檔案數: {stats['total_files']:,}
- 成功處理: {stats['successful_files']:,} ({stats['successful_files']/stats['total_files']*100:.1f}%)
- 處理失敗: {stats['failed_files']:,} ({stats['failed_files']/stats['total_files']*100:.1f}%)

### 年度分布
"""
    
    # 年度分布
    for year, count in sorted(stats['year_distribution'].items()):
        percentage = count / stats['total_files'] * 100
        report += f"- {year}年: {count:,} 件 ({percentage:.1f}%)\n"
    
    report += "\n### 案件類型分布\n"
    # 案件類型分布 (前10名)
    for case_type, count in stats['case_types'].most_common(10):
        percentage = count / stats['total_files'] * 100
        report += f"- {case_type}: {count:,} 件 ({percentage:.1f}%)\n"
    
    if len(stats['case_types']) > 10:
        report += f"- 其他 {len(stats['case_types']) - 10} 種類型...\n"
    
    report += "\n### 法院分布\n"
    for court, count in sorted(stats['court_types'].items()):
        percentage = count / stats['total_files'] * 100
        report += f"- {court}: {count:,} 件 ({percentage:.1f}%)\n"
    
    # 章節統計
    report += "\n## 📄 章節結構分析\n"
    
    section_names = ['header', 'main_text', 'facts', 'reasons', 'facts_and_reasons', 'footer']
    for section in section_names:
        if section in stats['section_stats']:
            lines = stats['section_stats'][section]
            present_count = stats['section_presence'][section]
            empty_count = stats['empty_sections'][section]
            
            if lines:
                avg_lines = sum(lines) / len(lines)
                max_lines = max(lines)
                min_lines = min(lines)
                median_lines = sorted(lines)[len(lines)//2]
                
                # 為合併格式添加特殊說明
                section_display = section.upper()
                if section == 'facts_and_reasons':
                    section_display = "FACTS_AND_REASONS (合併格式)"
                
                report += f"\n### {section_display} 章節\n"
                report += f"- 存在率: {present_count:,}/{stats['successful_files']:,} ({present_count/stats['successful_files']*100:.1f}%)\n"
                report += f"- 空章節: {empty_count:,} ({empty_count/stats['successful_files']*100:.1f}%)\n"
                report += f"- 平均行數: {avg_lines:.1f}\n"
                report += f"- 中位數行數: {median_lines}\n"
                report += f"- 行數範圍: {min_lines} - {max_lines}\n"
    
    # 符號化分類統計 - 新的清晰分類系統
    symbol_categories = stats.get('symbol_categories', {})
    total_processed = stats['successful_files']
    
    report += "\n## 🔤 符號化文件分類統計\n"
    report += "### 📊 符號說明\n"
    report += "- **M**: 主文 (Main text)\n"
    report += "- **F**: 事實 (Facts)\n" 
    report += "- **R**: 理由 (Reasons)\n"
    report += "- **S**: 事實及理由合併章節 (Simplified)\n"
    report += "- **D1**: 包含1個日期\n"
    report += "- **D2**: 包含2個或以上日期\n"
    report += "- **N**: 無有效內容\n"
    report += "- **ERROR**: 處理失敗\n\n"
    
    # 符號分類統計
    if symbol_categories:
        report += "### 📋 文件分類結果\n"
        
        # 按文件數量排序顯示
        sorted_symbols = sorted(symbol_categories.items(), key=lambda x: len(x[1]), reverse=True)
        
        for symbol, files in sorted_symbols:
            count = len(files)
            percentage = count / total_processed * 100 if total_processed > 0 else 0
            
            # 解釋符號含義
            description = ""
            if symbol == "ERROR":
                description = "處理失敗"
            elif symbol == "N":
                description = "無有效內容"
            else:
                parts = []
                if "M" in symbol:
                    parts.append("主文")
                if "S" in symbol:
                    parts.append("事實及理由合併")
                elif "F" in symbol:
                    parts.append("事實") 
                if "R" in symbol and "S" not in symbol:
                    parts.append("理由")
                if "D2" in symbol:
                    parts.append("多個日期")
                elif "D1" in symbol:
                    parts.append("單個日期")
                description = "+".join(parts) if parts else "其他"
            
            report += f"- **{symbol}**: {count:,} 份 ({percentage:.1f}%) - {description}\n"
            
            # 顯示前3個文件例子
            if count > 0 and symbol != "ERROR":
                report += f"  例子: "
                examples = [f['filename'] for f in files[:3]]
                report += ", ".join(examples)
                if count > 3:
                    report += f" ... (還有{count-3}個)"
                report += "\n"
    
    # 合併格式vs分離格式統計 - 保持向後兼容
    report += "\n## 🔄 文件分類統計 (精確計算)\n"
    
    report += "\n## 🔄 傳統分類統計 (基於符號分析)\n"
    
    # 基於符號重新計算傳統分類
    combined_format = []    # 包含FR的文件
    separated_format = []   # 包含MFR或MF但不包含FR的文件  
    procedural_only = []    # 只包含R的文件
    main_text_only = []     # 只包含M的文件
    facts_only = []         # 只包含F的文件
    incomplete = []         # N或其他異常組合
    
    for symbol, files in symbol_categories.items():
        if symbol == "ERROR":
            continue
        elif "S" in symbol:
            combined_format.extend(files)
        elif ("M" in symbol and "F" in symbol and "R" in symbol) or ("M" in symbol and "F" in symbol):
            separated_format.extend(files)
        elif symbol == "R" or (symbol.startswith("R") and "M" not in symbol and "F" not in symbol):
            procedural_only.extend(files)
        elif symbol == "M" or (symbol.startswith("M") and "F" not in symbol and "R" not in symbol):
            main_text_only.extend(files)
        elif symbol == "F" or (symbol.startswith("F") and "M" not in symbol and "R" not in symbol):
            facts_only.extend(files)
        else:
            incomplete.extend(files)
    
    report += f"### 📊 傳統分類映射\n"
    report += f"- 🔗 **合併格式** (包含S): {len(combined_format):,} 份 ({len(combined_format)/total_processed*100:.1f}%)\n"
    report += f"- ✂️ **分離格式** (MFR/MF): {len(separated_format):,} 份 ({len(separated_format)/total_processed*100:.1f}%)\n"
    report += f"- ⚖️ **程序性案件** (僅R): {len(procedural_only):,} 份 ({len(procedural_only)/total_processed*100:.1f}%)\n"
    report += f"- 📄 **僅有主文** (僅M): {len(main_text_only):,} 份 ({len(main_text_only)/total_processed*100:.1f}%)\n"
    report += f"- 📝 **僅有事實** (僅F): {len(facts_only):,} 份 ({len(facts_only)/total_processed*100:.1f}%)\n"
    report += f"- ❓ **其他格式** (N等): {len(incomplete):,} 份 ({len(incomplete)/total_processed*100:.1f}%)\n"
    
    
    # 計算總分類數進行驗證
    total_processed = stats['successful_files']
    categorized_total = (
        len(combined_format) + 
        len(separated_format) + 
        len(procedural_only) + 
        len(main_text_only) + 
        len(facts_only) + 
        len(incomplete)
    )
    
    report += f"\n### ✅ 分類驗證\n"
    report += f"- 成功處理的文件: {total_processed:,}\n"
    report += f"- 已分類的文件: {categorized_total:,}\n"
    report += f"- 分類完整性: {'✅ 完整' if categorized_total == total_processed else '❌ 有遺漏'}\n"
    
    # 詳細的章節存在統計
    report += f"\n### 📋 章節存在統計\n"
    detailed = stats['detailed_stats']
    report += f"- 有標題章節: {detailed['has_header']:,} 份 ({detailed['has_header']/total_processed*100:.1f}%)\n"
    report += f"- 有主文章節: {detailed['has_main_text']:,} 份 ({detailed['has_main_text']/total_processed*100:.1f}%)\n"
    report += f"- 有事實章節: {detailed['has_facts']:,} 份 ({detailed['has_facts']/total_processed*100:.1f}%)\n"
    report += f"- 有理由章節: {detailed['has_reasons']:,} 份 ({detailed['has_reasons']/total_processed*100:.1f}%)\n"
    report += f"- 有合併章節: {detailed['has_facts_and_reasons']:,} 份 ({detailed['has_facts_and_reasons']/total_processed*100:.1f}%)\n"
    report += f"- 有結尾章節: {detailed['has_footer']:,} 份 ({detailed['has_footer']/total_processed*100:.1f}%)\n"
    
    # 各符號類別的詳細信息
    if symbol_categories:
        # 找出最常見的符號組合
        sorted_symbols = sorted(symbol_categories.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 顯示前3個最常見符號的詳細信息
        for symbol, files in sorted_symbols[:3]:
            if symbol == "ERROR" or not files:
                continue
                
            report += f"\n### 📊 '{symbol}' 類別詳情\n"
            report += f"- 文件數量: {len(files)} 份\n"
            
            # 計算平均行數
            if files and 'main_text_lines' in files[0]:
                avg_main = sum(f.get('main_text_lines', 0) for f in files) / len(files)
                avg_facts = sum(f.get('facts_lines', 0) for f in files) / len(files)
                avg_reasons = sum(f.get('reasons_lines', 0) for f in files) / len(files)
                avg_combined = sum(f.get('combined_lines', 0) for f in files) / len(files)
                
                report += f"- 平均主文行數: {avg_main:.1f}\n"
                report += f"- 平均事實行數: {avg_facts:.1f}\n"
                report += f"- 平均理由行數: {avg_reasons:.1f}\n"
                if avg_combined > 0:
                    report += f"- 平均合併行數: {avg_combined:.1f}\n"
            
            # 顯示例子檔案
            report += f"- 例子檔案:\n"
            for i, file_info in enumerate(files[:3], 1):
                report += f"  {i}. {file_info['filename']}\n"

    # 日期提取統計
    report += "\n## 📅 日期提取分析\n"
    date_stats = stats['date_extraction_stats']
    report += f"- 包含日期的檔案: {date_stats['files_with_dates']:,}/{stats['successful_files']:,} ({date_stats['files_with_dates']/stats['successful_files']*100:.1f}%)\n"
    report += f"- 總提取日期數: {date_stats['total_dates_found']:,}\n"
    
    if date_stats['date_patterns']:
        report += "\n### 日期分布 (前10名)\n"
        for date_pattern, count in date_stats['date_patterns'].most_common(10):
            report += f"- {date_pattern}: {count:,} 次\n"
    
    # 檔案大小統計
    if stats['file_sizes']:
        report += "\n## 💾 檔案大小分析\n"
        sizes = stats['file_sizes']
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        min_size = min(sizes)
        median_size = sorted(sizes)[len(sizes)//2]
        
        report += f"- 平均大小: {avg_size/1024:.1f} KB\n"
        report += f"- 中位數大小: {median_size/1024:.1f} KB\n"
        report += f"- 大小範圍: {min_size/1024:.1f} KB - {max_size/1024:.1f} KB\n"
    
    # 錯誤統計
    if stats['processing_errors']:
        report += f"\n## ⚠️ 處理錯誤 ({len(stats['processing_errors'])} 件)\n"
        error_types = Counter()
        for error in stats['processing_errors']:
            error_type = error['error'].split(':')[0]
            error_types[error_type] += 1
        
        for error_type, count in error_types.most_common():
            report += f"- {error_type}: {count} 件\n"
        
        if len(stats['processing_errors']) <= 10:
            report += "\n### 詳細錯誤列表\n"
            for error in stats['processing_errors']:
                report += f"- {error['file']}: {error['error']}\n"
    
    # 模式識別統計
    report += "\n## 🔍 模式識別結果\n"
    patterns = find_section_patterns()
    for name, pattern in patterns.items():
        if hasattr(pattern, 'pattern'):
            report += f"- {name}: `{pattern.pattern}`\n"
        else:
            report += f"- {name}: `{pattern}`\n"
    
    return report

def generate_detailed_category_report(stats):
    """生成詳細的符號分類報告"""
    
    symbol_categories = stats.get('symbol_categories', {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 為每個符號類別生成詳細報告
    for symbol, files in symbol_categories.items():
        if not files:
            continue
            
        report_filename = output_dir / f"symbol_{symbol}_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"# 符號 '{symbol}' 類別詳細報告\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文件數量: {len(files)}\n\n")
            
            # 符號含義說明
            f.write("## 符號含義\n")
            meanings = []
            if "M" in symbol:
                meanings.append("M = 主文 (Main text)")
            if "F" in symbol:
                meanings.append("F = 事實 (Facts)")
            if "R" in symbol:
                meanings.append("R = 理由 (Reasons)")
            if "S" in symbol:
                meanings.append("S = 事實及理由合併 (Simplified)")
            if "D1" in symbol:
                meanings.append("D1 = 包含1個日期")
            if "D2" in symbol:
                meanings.append("D2 = 包含2個或以上日期")
            if symbol == "N":
                meanings.append("N = 無有效內容")
            if symbol == "ERROR":
                meanings.append("ERROR = 處理失敗")
            
            for meaning in meanings:
                f.write(f"- {meaning}\n")
            f.write("\n")
            
            # 文件列表
            f.write("## 文件列表\n")
            for i, file_info in enumerate(files, 1):
                f.write(f"{i:4d}. {file_info['filename']}\n")
                
                # 顯示章節統計
                if 'main_text_lines' in file_info:
                    f.write(f"      主文: {file_info['main_text_lines']} 行\n")
                if 'facts_lines' in file_info:
                    f.write(f"      事實: {file_info['facts_lines']} 行\n")
                if 'reasons_lines' in file_info:
                    f.write(f"      理由: {file_info['reasons_lines']} 行\n")
                if 'combined_lines' in file_info:
                    f.write(f"      合併: {file_info['combined_lines']} 行\n")
                if 'dates_count' in file_info:
                    f.write(f"      日期: {file_info['dates_count']} 個\n")
                if 'header_lines' in file_info:
                    f.write(f"      標題: {file_info['header_lines']} 行\n")
                if 'footer_lines' in file_info:
                    f.write(f"      結尾: {file_info['footer_lines']} 行\n")
                
                # 如果是錯誤文件，顯示錯誤信息
                if 'error' in file_info:
                    f.write(f"      錯誤: {file_info['error']}\n")
                
                f.write("\n")
        
        print(f"📄 符號 '{symbol}' 詳細報告已保存至: {report_filename}")
    
    # 生成符號統計摘要
    summary_filename = f"symbol_summary_{timestamp}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("# 符號分類統計摘要\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 符號說明\n")
        f.write("- M: 主文 (Main text)\n")
        f.write("- F: 事實 (Facts)\n")
        f.write("- R: 理由 (Reasons)\n")
        f.write("- S: 事實及理由合併章節 (Simplified)\n")
        f.write("- D1: 包含1個日期\n")
        f.write("- D2: 包含2個或以上日期\n")
        f.write("- N: 無有效內容\n")
        f.write("- ERROR: 處理失敗\n\n")
        
        f.write("## 分類統計\n")
        sorted_symbols = sorted(symbol_categories.items(), key=lambda x: len(x[1]), reverse=True)
        total_files = sum(len(files) for files in symbol_categories.values())
        
        for symbol, files in sorted_symbols:
            count = len(files)
            percentage = count / total_files * 100 if total_files > 0 else 0
            f.write(f"{symbol:>8}: {count:5d} 份 ({percentage:5.1f}%)\n")
        
        f.write(f"\n總計: {total_files:5d} 份\n")
    
    print(f"📊 符號統計摘要已保存至: {summary_filename}")

def main():
    """主函數"""
    print("🚀 L-chunk 綜合分析開始")
    print("=" * 60)
    
    # 執行分析
    stats = analyze_filtered_dataset()
    
    if stats:
        # 生成報告
        report = generate_comprehensive_report(stats)
        
        # 保存報告
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_filename = output_dir / f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 分析完成！報告已保存至: {report_filename}")
        
        # 保存 JSON 格式的原始數據
        json_filename = output_dir / f"analysis_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 轉換不能序列化的數據
        json_stats = dict(stats)
        json_stats['section_stats'] = {k: v for k, v in stats['section_stats'].items()}
        json_stats['case_types'] = dict(stats['case_types'])
        json_stats['year_distribution'] = dict(stats['year_distribution'])
        json_stats['court_types'] = dict(stats['court_types'])
        json_stats['section_presence'] = dict(stats['section_presence'])
        json_stats['empty_sections'] = dict(stats['empty_sections'])
        json_stats['date_extraction_stats']['date_patterns'] = dict(stats['date_extraction_stats']['date_patterns'])
        
        # 新增：符號分類數據和文件分類數據
        json_stats['symbol_categories'] = stats.get('symbol_categories', {})
        json_stats['file_categories'] = stats.get('file_categories', {})
        json_stats['detailed_stats'] = stats['detailed_stats']
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 原始數據已保存至: {json_filename}")
        
        # 生成詳細分類報告
        generate_detailed_category_report(stats)
        
        # 簡要摘要
        print(f"\n📋 處理摘要:")
        print(f"   總檔案: {stats['total_files']:,}")
        print(f"   成功率: {stats['successful_files']/stats['total_files']*100:.1f}%")
        print(f"   處理時間: {stats['processing_time']:.1f} 秒")
        print(f"   平均速度: {stats['total_files']/stats['processing_time']:.1f} 檔案/秒")
    
    else:
        print("❌ 分析失敗")

if __name__ == "__main__":
    main()