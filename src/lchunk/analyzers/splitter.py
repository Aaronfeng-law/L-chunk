#!/usr/bin/env python3
"""
Judgment Document Splitter
Splits judgment documents based on structural patterns found in lines 40, 58, 100, 1504, 1514
"""

import json
import sys
import os
import re
from pathlib import Path

def normalize_text(text):
    """Remove spaces and \r\n from text for pattern matching"""
    return re.sub(r'[\s\r\n]', '', text)

def find_section_patterns():
    """Define the patterns found in the specified lines"""
    patterns = {
        'main_text': '主文',              # Line 40: 主文
        'facts': '事實',                  # Line 58: 事實  
        'reasons': '理由',                # Line 100: 理由
        'facts_and_reasons_pattern': re.compile(r'^\s*事實\s*[及與和]\s*理由\s*$'),  # Flexible combined format pattern
        'date_pattern': re.compile(r'中\s*華\s*民\s*國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日'),  # ROC date pattern
        'date_pattern_strict': re.compile(r'中\s*華\s*民\s*國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'),  # More strict
    }
    return patterns

def extract_dates_from_text(text):
    """Extract ROC dates from text using improved two-step process"""
    dates = []
    
    # Step 1: Handle different line separators
    if isinstance(text, list):
        lines = text
    else:
        # Try different line separators
        if '\r\n' in text:
            lines = text.split('\r\n')
        elif '\n' in text:
            lines = text.split('\n')
        else:
            lines = [text]
    
    # Step 2: Process each line
    for line in lines:
        # Check for date components (more flexible)
        if ('中' in line and '華' in line and '民' in line and '國' in line and 
            '年' in line and '月' in line and '日' in line):
            
            # Step 3: Aggressive cleaning of all types of spaces
            cleaned_line = line
            # Replace all common space characters
            space_chars = ['\u3000', '\u0020', '\u00A0', '\u2000', '\u2001', '\u2002', 
                          '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', 
                          '\u2009', '\u200A', '\u200B', '\u202F', '\u205F', '\u3000']
            for space_char in space_chars:
                cleaned_line = cleaned_line.replace(space_char, ' ')
            
            # Normalize all whitespace to single spaces, then remove
            cleaned_line = re.sub(r'\s+', '', cleaned_line)
            
            # Step 4: Use multiple regex patterns with flexible spacing on original line
            patterns = [
                r'中\s*華\s*民\s*國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日',  # Flexible spacing on original
                r'中華民國(\d+)年(\d+)月(\d+)日',  # Standard format on cleaned
                r'民國(\d+)年(\d+)月(\d+)日',      # Short format on cleaned
            ]
            
            # Try patterns on both original (with \s*) and cleaned line
            test_lines = [line, cleaned_line]
            
            for test_line in test_lines:
                for pattern in patterns:
                    match = re.search(pattern, test_line)
                    if match:
                        year, month, day = match.groups()
                        dates.append({
                            'year': year,
                            'month': month, 
                            'day': day,
                            'full_match': f"中華民國{year}年{month}月{day}日",
                            'original_line': line,
                            'cleaned_line': cleaned_line
                        })
                        break  # Only take first match per line
                if dates and dates[-1]['original_line'] == line:
                    break  # Found a match for this line, move to next line
    
    return dates

def split_judgment(jfull_content):
    """
    Split judgment content based on identified patterns
    
    Args:
        jfull_content (str): Full judgment text with \r\n separators
        
    Returns:
        dict: Dictionary with split sections
    """
    patterns = find_section_patterns()
    
    # Split into lines and clean each line for pattern matching
    lines = jfull_content.split('\r\n')
    
    sections = {
        'header': [],
        'main_text': [],
        'facts': [],
        'reasons': [],
        'facts_and_reasons': [],  # New combined section
        'footer': []
    }
    
    current_section = 'header'
    
    for i, line in enumerate(lines):
        cleaned_line = normalize_text(line)
        
        # Check for section markers - 總是檢測重要章節，不受當前段落狀態影響
        if patterns['main_text'] in cleaned_line:
            current_section = 'main_text'
            sections[current_section].append(line)
        elif patterns['facts_and_reasons_pattern'].match(cleaned_line):
            # Handle combined facts and reasons format using flexible pattern
            current_section = 'facts_and_reasons'
            sections[current_section].append(line)
        elif patterns['facts'] in cleaned_line:
            current_section = 'facts' 
            sections[current_section].append(line)
        elif patterns['reasons'] in cleaned_line:
            current_section = 'reasons'
            sections[current_section].append(line)
        elif patterns['date_pattern_strict'].search(line) or patterns['date_pattern'].search(line):
            # Use strict regex first, then fallback to general pattern
            current_section = 'footer'
            sections[current_section].append(line)
        else:
            sections[current_section].append(line)
    
    # Post-process: if facts_and_reasons has content, populate facts and reasons from it
    if sections['facts_and_reasons']:
        # If we found combined facts_and_reasons, copy content to facts for compatibility
        if not sections['facts']:
            sections['facts'] = sections['facts_and_reasons'].copy()
        # Leave reasons empty for combined format to indicate it's merged
    
    return sections

def process_single_file(file_path):
    """Process a single JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'JFULL' not in data:
            return False, "No JFULL field found"
        
        sections = split_judgment(data['JFULL'])
        
        # Add metadata
        result = {
            'file_path': str(file_path),
            'jid': data.get('JID', 'Unknown'),
            'sections': sections,
            'section_stats': {
                section: len(content) for section, content in sections.items()
            }
        }
        
        return True, result
        
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def test_on_sample(verbose=False):
    """Test the splitter on sample files"""
    sample_dir = Path("data/sample")
    
    if not sample_dir.exists():
        print(f"Sample directory {sample_dir} not found")
        return
    
    print("Testing on sample files...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    results = []
    
    for json_file in sample_dir.glob("*.json"):
        success, result = process_single_file(json_file)
        
        if success:
            success_count += 1
            results.append(result)
            print(f"✅ {json_file.name}")
            print(f"   Sections: {result['section_stats']}")
            
            if verbose:
                print(f"   📝 Detailed sections for {json_file.name}:")
                for section_name, content in result['sections'].items():
                    if content:  # Only show non-empty sections
                        print(f"      {section_name}: {len(content)} lines")
                        # Show first few lines of each section
                        for i, line in enumerate(content[:3]):
                            print(f"        {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
                        if len(content) > 3:
                            print(f"        ... and {len(content) - 3} more lines")
                print()
        else:
            fail_count += 1
            print(f"❌ {json_file.name}: {result}")
    
    print("=" * 60)
    print(f"Sample Test Results:")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")
    
    # Save detailed results
    with open('sample_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': {
                'success': success_count,
                'failed': fail_count,
                'total': success_count + fail_count
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    return success_count, fail_count

def test_on_filtered(verbose=False):
    """Test the splitter on filtered files"""
    filtered_dir = Path("data/processed/filtered")
    
    if not filtered_dir.exists():
        print(f"Filtered directory {filtered_dir} not found")
        return
    
    print("Testing on filtered files...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    results = []
    
    json_files = list(filtered_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files[:10]:  # Test first 10 files
        success, result = process_single_file(json_file)
        
        if success:
            success_count += 1
            results.append(result)
            print(f"✅ {json_file.name}")
            print(f"   Sections: {result['section_stats']}")
            
            if verbose:
                print(f"   📝 Detailed sections for {json_file.name}:")
                for section_name, content in result['sections'].items():
                    if content:  # Only show non-empty sections
                        print(f"      {section_name}: {len(content)} lines")
                        # Show first few lines of each section
                        for i, line in enumerate(content[:3]):
                            print(f"        {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
                        if len(content) > 3:
                            print(f"        ... and {len(content) - 3} more lines")
                print()
        else:
            fail_count += 1
            print(f"❌ {json_file.name}: {result}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        verbose = "--verbose" in sys.argv
        
        if '--sample' in sys.argv:
            test_on_sample(verbose)
        elif '--filtered' in sys.argv:
            test_on_filtered(verbose)  
        
        elif '--patterns' in sys.argv:
            patterns = find_section_patterns()
            print("Identified patterns:")
            for name, pattern in patterns.items():
                if hasattr(pattern, 'pattern'):  # regex object
                    print(f"  {name}: {pattern.pattern}")
                else:
                    print(f"  {name}: '{pattern}'")
            
            # Test date extraction on sample
            print("\nTesting date extraction on sample line:")
            test_line = "中　　華　　民　　國　　114 　年　　1 　　月　　13　　日"
            dates = extract_dates_from_text(test_line)
            print(f"Test line: {test_line}")
            print(f"Extracted dates: {dates}")
        else:
            print("Usage: python judgment_splitter.py [--sample|--filtered|--patterns] [--verbose]")
            print("  --sample   : Test on sample files")
            print("  --filtered : Test on filtered files")
            print("  --patterns : Show detected patterns")
            print("  --verbose  : Show detailed section content")
    else:
        print("Usage: python judgment_splitter.py [--sample|--filtered|--patterns]")
        print("  --sample    Test on sample files")
        print("  --filtered  Test on filtered files")  
        print("  --patterns  Show identified patterns")

if __name__ == "__main__":
    main()
