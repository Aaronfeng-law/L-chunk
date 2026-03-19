#!/usr/bin/env python3
"""
<<<<<<< HEAD
終極嚴格格式層級檢測器
基於  "零容忍" 原則：
"規則要麼是絕對的，要麼就不是規則"

終極嚴格規則：
=======
嚴格格式層級檢測器
基於  "" 原則：
"規則要麼是絕對的，要麼就不是規則"

嚴格規則：
>>>>>>> master
1. 必須以 \r\n 開頭  
2. 只能是單個 Unicode 字符
3. 必須緊跟 、(頓號)
4. 字符必須在預定義範圍內（包含PUA）
5. 零例外，零妥協
"""

import json
import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
<<<<<<< HEAD

@dataclass
class UltraStrictMarker:
    """終極嚴格格式的層級標記"""
=======
from src.lchunk.pipeline import JudgmentArtifact, SymbolDetection

@dataclass
class UltraStrictMarker:
    """嚴格格式的層級標記"""
>>>>>>> master
    line_number: int
    symbol: str
    unicode_code: str
    category: str
    content: str
    is_pua: bool
    has_proper_newline: bool

class UltraStrictDetector:
<<<<<<< HEAD
    """終極嚴格格式檢測器"""
    
    def __init__(self):
        # 定義允許的單字符符號範圍（零容忍策略）
=======
    """嚴格格式檢測器"""
    
    def __init__(self):
        # 定義允許的單字符符號範圍（策略）
>>>>>>> master
        self.valid_symbol_ranges = {
            # 中文數字
            "中文數字": {
                "chars": {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"},
            },
            # 大寫中文數字
            "大寫中文數字": {
                "chars": {"壹", "貳", "參", "肆", "伍", "陸", "柒", "捌", "玖", "拾"},
            },
            # 天干
            "天干": {
                "chars": {"甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"},
            },
            # 地支
            "地支": {
                "chars": {"子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"},
            },
            # 圍字符號 - 帶圈數字
            "圍字數字": {
                "chars": {"㈠", "㈡", "㈢", "㈣", "㈤", "㈥", "㈦", "㈧", "㈨", "㈩"},
            },
            # 帶圈數字系列 - Unicode標準圈數字
            "帶圈數字": {
                "chars": {
                    # U+2460-U+2473: ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩ ⑪ ⑫ ⑬ ⑭ ⑮ ⑯ ⑰ ⑱ ⑲ ⑳
                    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", 
                    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
                    # U+24EA: ⓪ (圈零)
                    "⓪"
                },
            },
            # 圍字符號 - 帶圈括號數字
            "括號數字": {
                "chars": {"⑴", "⑵", "⑶", "⑷", "⑸", "⑹", "⑺", "⑻", "⑼", "⑽", "⑾", "⑿", "⒀", "⒁", "⒂", "⒃", "⒄", "⒅", "⒆", "⒇"},
            },
            # 全形數字
            "全形數字": {
                "chars": {"０", "１", "２", "３", "４", "５", "６", "７", "８", "９"},
            },
            # 羅馬數字
            "羅馬數字": {
                "chars": {"Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ"},
            }
        }
        
<<<<<<< HEAD
        # PUA符號分組 - 基於格式精確分類（零容忍分組）
        self.pua_symbol_groups = self._init_pua_groups()
        
        # PUA範圍 (Private Use Area) - 終極精確定義
=======
        # PUA符號分組 - 基於格式精確分類（分組）
        self.pua_symbol_groups = self._init_pua_groups()
        
        # PUA範圍 (Private Use Area) - 精確定義
>>>>>>> master
        self.pua_ranges = [
            (0xE000, 0xF8FF),   # Basic Multilingual Plane PUA
            (0xF0000, 0xFFFFD), # Supplementary PUA-A
            (0x100000, 0x10FFFD) # Supplementary PUA-B
        ]
        
        # 創建所有有效字符的集合（用於快速查找）
        self.all_valid_chars = set()
        for category_info in self.valid_symbol_ranges.values():
            self.all_valid_chars.update(category_info["chars"])
    
    def _init_pua_groups(self) -> Dict:
        """初始化PUA符號分組 - 精確分類
        
        分組原則：
        1. 全形 vs 半形分開
        2. 小寫 vs 大寫分開  
        3. 圈圈、括弧、句號、頓號分開
        4. 羅馬數字、阿拉伯數字、中文數字分開
        5. 每組都有明確的層級定義
        """
        return {
            # === 羅馬數字系列 ===
            "PUA_羅馬數字": {
                "ranges": [(0xF6C5, 0xF6CE)],  # F6C5~F6CE
                "description": "羅馬數字（PUA定義）",
<<<<<<< HEAD
=======
                "format_type": "羅馬數字",
>>>>>>> master
            },
            
            # === 半形阿拉伯數字系列 ===
            "PUA_半形阿拉伯數字_括弧": {
                "ranges": [
                    (0xF6BB, 0xF6C4),  # F6BB~F6C4: 半形阿拉伯數字帶括弧(1~10)
                    (0xF4FB, 0xF4FF),  # F4FB~F4FF: 半形阿拉伯數字括弧(11~15)
                    (0xF4EA, 0xF4EB)   # F4EA~F4EB: 半形阿拉伯數字括弧(16~17)
                ],
                "description": "半形阿拉伯數字括弧系列",
<<<<<<< HEAD
=======
                "format_type": "半形括弧",
>>>>>>> master
            },
            
            # === 全形阿拉伯數字系列 ===
            "PUA_全形阿拉伯數字_原數字": {
                "ranges": [(0xF5E9, 0xF64C)],  # F5E9~F64C: 全形阿拉伯數字（1~100）
                "description": "全形阿拉伯數字原型",
<<<<<<< HEAD
=======
                "format_type": "全形原型",
>>>>>>> master
            },
            
            "PUA_全形阿拉伯數字_句號": {
                "ranges": [(0xF585, 0xF5E8)],  # F585~F5E8: 全形阿拉伯數字句號（1~100）
                "description": "全形阿拉伯數字句號",
<<<<<<< HEAD
=======
                "format_type": "全形句號",
>>>>>>> master
            },
            
            "PUA_全形阿拉伯數字_圈圈": {
                "ranges": [
                    (0xF6B1, 0xF6BA),  # F6B1~F6BA: 阿拉伯數字with圈圈
                    (0xF521, 0xF584)   # F521~F584: 全形阿拉伯數字with圈圈
                ],
                "description": "全形阿拉伯數字圈圈",
<<<<<<< HEAD
=======
                "format_type": "全形圈圈",
>>>>>>> master
            },
            
            "PUA_全形阿拉伯數字_括弧": {
                "ranges": [
                    (0xF514, 0xF520),  # F514~F520: 全形阿拉伯數字括弧（78~90）
                    (0xF500, 0xF501),  # F500~F501: 全形阿拉伯數字括弧（91~92）
                    (0xF4F9, 0xF4FA),  # F4F9~F4FA: 全形阿拉伯數字括弧（93~94）
                    (0xF4EF, 0xF4F4)   # F4EF~F4F4: 全形阿拉伯數字括弧（95~100）
                ],
                "description": "全形阿拉伯數字括弧",
<<<<<<< HEAD
=======
                "format_type": "全形括弧",
>>>>>>> master
            },
            
            # === 小寫中文數字系列 ===
            "PUA_小寫中文數字_頓號": {
                "ranges": [(0xF57F, 0xF6B0)],  # F57F~F6B0: 小寫國字數字頓號，二位數垂直排列（1~50）
                "description": "小寫中文數字頓號（二位數垂直）",
<<<<<<< HEAD
=======
                "format_type": "中文頓號",
>>>>>>> master
            },
            
            "PUA_小寫中文數字_括弧": {
                "ranges": [
                    (0xF64D, 0xF67E),  # F64D~F67E: 小寫國字數字括弧，二位數垂直排列（1~50）
                    (0xF502, 0xF509)   # F502~F509: 小寫國字數字括弧，二位數垂直排列（83~87）
                ],
                "description": "小寫中文數字括弧（二位數垂直）",
<<<<<<< HEAD
=======
                "format_type": "中文括弧",
>>>>>>> master
            },
            
            # === 天干系列 ===
            "PUA_天干_括弧": {
                "ranges": [(0xF50A, 0xF513)],  # F50A~F513: 天干括弧
                "description": "天干括弧",
<<<<<<< HEAD
=======
                "format_type": "天干括弧",
>>>>>>> master
            }
        }
    
    def is_pua_character(self, char: str) -> bool:
<<<<<<< HEAD
        """檢查是否為PUA字符（終極精確）"""
=======
        """檢查是否為PUA字符"""
>>>>>>> master
        if not char:
            return False
        
        code_point = ord(char)
        for start, end in self.pua_ranges:
            if start <= code_point <= end:
                return True
        return False
    
    def is_valid_symbol(self, char: str) -> Tuple[bool, str]:
<<<<<<< HEAD
        """檢查字符是否在有效範圍內（零容忍）"""
=======
        """檢查字符是否在有效範圍內"""
>>>>>>> master
        # 檢查預定義範圍
        for category, info in self.valid_symbol_ranges.items():
            if char in info["chars"]:
                return True, category
        
        # 檢查PUA精確分組
<<<<<<< HEAD
        pua_group = self.get_pua_group(char)
=======
            pua_group = self.get_pua_group(char)
>>>>>>> master
        if pua_group:
            return True, pua_group
        
        return False, "無效字符"
    
    def get_pua_group(self, char: str) -> str:
<<<<<<< HEAD
        """獲取PUA字符的精確分組（零容忍分類）
=======
        """獲取PUA字符的精確分組（分類）
>>>>>>> master
        
        特殊合併規則：PUA_全形阿拉伯數字_原數字 → 全形數字
        """
        if not char:
            return ""
        
        code_point = ord(char)
        
        # 檢查每個PUA分組
        for group_name, group_info in self.pua_symbol_groups.items():
            for start, end in group_info["ranges"]:
                if start <= code_point <= end:
                    # 合併：PUA全形數字與標準全形數字統一
                    if group_name == "PUA_全形阿拉伯數字_原數字":
                        return "全形數字"  # 合併到標準全形數字類別
                    return group_name
        
        # 如果在PUA範圍內但不在分組中，返回通用PUA標記
        if self.is_pua_character(char):
            return "PUA_未分組"
        
        return ""
    
    def get_symbol_level(self, char: str, category: str) -> int:
        """獲取符號的層級（基於精確分組）
        
        特殊處理：合併後的全形數字類別統一層級
<<<<<<< HEAD
        """
        # 預定義範圍的層級
        if category in self.valid_symbol_ranges:
            return self.valid_symbol_ranges[category]["level"]
        
        # 合併處理：如果是合併後的全形數字，使用預定義的層級
        if category == "全形數字" and category in self.valid_symbol_ranges:
            return self.valid_symbol_ranges[category]["level"]
        
        # PUA分組的層級
        if category in self.pua_symbol_groups:
            return self.pua_symbol_groups[category]["level"]
=======
        level 欄位為選用，未設時預設為 0（實際層級由 AdaptiveHybridDetector 動態學習決定）
        """
        # 預定義範圍的層級
        if category in self.valid_symbol_ranges:
            return self.valid_symbol_ranges[category].get("level", 0)
        
        # 合併處理：如果是合併後的全形數字，使用預定義的層級
        if category == "全形數字" and category in self.valid_symbol_ranges:
            return self.valid_symbol_ranges[category].get("level", 0)
        
        # PUA分組的層級
        if category in self.pua_symbol_groups:
            return self.pua_symbol_groups[category].get("level", 0)
>>>>>>> master
        
        # 默認層級
        return 0
    
    def detect_ultra_strict_markers(self, text_lines: List[str]) -> List[UltraStrictMarker]:
<<<<<<< HEAD
        """終極嚴格檢測層級標記"""
        markers = []
        
        # 終極嚴格的正則表達式：單個字符 + 頓號，無任何例外
=======
        """嚴格檢測層級標記"""
        markers = []
        
        # 嚴格的正則表達式：單個字符 + 頓號，無任何例外
>>>>>>> master
        ultra_strict_pattern = r'^(.{1})、(.*)$'
        
        for line_num, line in enumerate(text_lines, 1):
            stripped_line = line.strip()
            
<<<<<<< HEAD
            # 跳過空行（零容忍：空行不是有效標記）
            if not stripped_line:
                continue
            
            # 檢查是否符合終極嚴格格式
=======
            # 跳過空行（：空行不是有效標記）
            if not stripped_line:
                continue
            
            # 檢查是否符合嚴格格式
>>>>>>> master
            match = re.match(ultra_strict_pattern, stripped_line)
            if match:
                symbol = match.group(1)
                content_after_comma = match.group(2)
                
<<<<<<< HEAD
                # 終極嚴格要求1：必須是單個字符（已由正則保證）
                # 終極嚴格要求2：字符必須在有效範圍內
                is_valid, category = self.is_valid_symbol(symbol)
                
                if is_valid:
                    # 終極嚴格要求3：檢查前一行是否以\r結尾（如果不是第一行）
=======
                # 嚴格要求1：必須是單個字符（已由正則保證）
                # 嚴格要求2：字符必須在有效範圍內
                is_valid, category = self.is_valid_symbol(symbol)
                
                if is_valid:
                    # 嚴格要求3：檢查前一行是否以\r結尾（如果不是第一行）
>>>>>>> master
                    has_proper_newline = True
                    if line_num > 1:
                        prev_line = text_lines[line_num - 2]  # 0-indexed
                        has_proper_newline = prev_line.endswith('\r')
                    
<<<<<<< HEAD
                    # 終極嚴格要求4：只有符合所有條件的才被接受（零妥協）
=======
                    # 嚴格要求4：只有符合所有條件的才被接受（零妥協）
>>>>>>> master
                    if has_proper_newline:
                        marker = UltraStrictMarker(
                            line_number=line_num,
                            symbol=symbol,
                            unicode_code=f"U+{ord(symbol):04X}",
                            category=category,
                            content=stripped_line,
                            is_pua=self.is_pua_character(symbol),
                            has_proper_newline=has_proper_newline
                        )
                        markers.append(marker)
        
        return markers
    
    def analyze_ultra_strict_structure(self, markers: List[UltraStrictMarker]) -> Dict:
<<<<<<< HEAD
        """分析終極嚴格檢測的結構"""
=======
        """分析嚴格檢測的結構"""
>>>>>>> master
        analysis = {
            "total_markers": len(markers),
            "by_category": {},
            "by_pua_group": {},
            "by_format_type": {},
            "by_level": {},
            "pua_count": 0,
            "unicode_distribution": {},
            "newline_compliance": 0,
            "structure": []
        }
        
        # 按類別統計
        for marker in markers:
            if marker.category not in analysis["by_category"]:
                analysis["by_category"][marker.category] = []
            
            analysis["by_category"][marker.category].append({
                "line": marker.line_number,
                "symbol": marker.symbol,
                "unicode": marker.unicode_code,
                "is_pua": marker.is_pua,
                "has_proper_newline": marker.has_proper_newline
            })
            
            # PUA計數和精確分組統計
            if marker.is_pua:
                analysis["pua_count"] += 1
                
                # PUA精確分組統計
                if marker.category not in analysis["by_pua_group"]:
                    analysis["by_pua_group"][marker.category] = {
                        "count": 0,
                        "description": "",
                        "symbols": []
                    }
                
                analysis["by_pua_group"][marker.category]["count"] += 1
                analysis["by_pua_group"][marker.category]["symbols"].append({
                    "line": marker.line_number,
                    "symbol": marker.symbol,
                    "unicode": marker.unicode_code
                })
                
                # 獲取PUA分組的詳細信息
                if marker.category in self.pua_symbol_groups:
                    group_info = self.pua_symbol_groups[marker.category]
<<<<<<< HEAD
                    analysis["by_pua_group"][marker.category]["format_type"] = group_info["format_type"]
                    analysis["by_pua_group"][marker.category]["level"] = group_info["level"]
                    analysis["by_pua_group"][marker.category]["description"] = group_info["description"]
=======
                    analysis["by_pua_group"][marker.category]["format_type"] = group_info.get("format_type", "")
                    analysis["by_pua_group"][marker.category]["level"] = group_info.get("level", 0)
                    analysis["by_pua_group"][marker.category]["description"] = group_info.get("description", "")
>>>>>>> master
            
            # Unicode分布
            if marker.unicode_code not in analysis["unicode_distribution"]:
                analysis["unicode_distribution"][marker.unicode_code] = 0
            analysis["unicode_distribution"][marker.unicode_code] += 1
            
            # 換行符合性
            if marker.has_proper_newline:
                analysis["newline_compliance"] += 1
        
        # 按格式類型統計
        for marker in markers:
            if marker.is_pua and marker.category in self.pua_symbol_groups:
<<<<<<< HEAD
                format_type = self.pua_symbol_groups[marker.category]["format_type"]
                if format_type not in analysis["by_format_type"]:
                    analysis["by_format_type"][format_type] = 0
                analysis["by_format_type"][format_type] += 1
=======
                format_type = self.pua_symbol_groups[marker.category].get("format_type", "")
                if format_type and format_type not in analysis["by_format_type"]:
                    analysis["by_format_type"][format_type] = 0
                if format_type:
                    analysis["by_format_type"][format_type] += 1
>>>>>>> master
        
        # 按層級統計（包含PUA）
        for marker in markers:
            level = self.get_symbol_level(marker.symbol, marker.category)
            if level not in analysis["by_level"]:
                analysis["by_level"][level] = 0
            analysis["by_level"][level] += 1
        
        # 結構化表示
        for marker in markers:
            level = self.get_symbol_level(marker.symbol, marker.category)
            format_type = ""
            if marker.is_pua and marker.category in self.pua_symbol_groups:
<<<<<<< HEAD
                format_type = self.pua_symbol_groups[marker.category]["format_type"]
=======
                format_type = self.pua_symbol_groups[marker.category].get("format_type", "")
>>>>>>> master
            
            analysis["structure"].append({
                "line": marker.line_number,
                "symbol": marker.symbol,
                "unicode": marker.unicode_code,
                "category": marker.category,
                "is_pua": marker.is_pua,
<<<<<<< HEAD
=======
                "level": level,
                "format_type": format_type,
>>>>>>> master
                "content": marker.content[:80] + "..." if len(marker.content) > 80 else marker.content,
                "newline_ok": marker.has_proper_newline
            })
        
        return analysis
    
    def generate_ultra_strict_report(self, markers: List[UltraStrictMarker], analysis: Dict) -> str:
<<<<<<< HEAD
        """生成終極嚴格檢測報告"""
=======
    
        """生成嚴格檢測報告"""
>>>>>>> master
        newline_compliance_rate = (analysis["newline_compliance"] / len(markers) * 100) if markers else 0
        
        report = f"""
============================================================
<<<<<<< HEAD
⚡ 終極嚴格格式層級檢測報告 (精確分組)
============================================================

🔒 終極嚴格規則 (零容忍):
=======
⚡ 嚴格格式層級檢測報告 (精確分組)
============================================================

🔒 嚴格規則 ():
>>>>>>> master
  ✓ 必須以 \\r\\n 開頭
  ✓ 只能是單個 Unicode 字符  
  ✓ 必須緊跟 、(頓號)
  ✓ 字符必須在預定義範圍內（含PUA精確分組）
<<<<<<< HEAD
  ✓ 零例外，零妥協

📊 檢測結果:
  符合終極嚴格格式的標記: {len(markers)} 個
=======
  
📊 檢測結果:
  符合嚴格格式的標記: {len(markers)} 個
>>>>>>> master
  PUA字符標記: {analysis['pua_count']} 個
  換行符合性: {analysis['newline_compliance']}/{len(markers)} ({newline_compliance_rate:.1f}%)
  
📈 按類別分布:
"""
        
        for category, items in analysis["by_category"].items():
            is_pua = category.startswith("PUA_")
            pua_mark = " (PUA)" if is_pua else ""
            newline_ok = sum(1 for item in items if item["has_proper_newline"])
            report += f"  {category:35s}{pua_mark}: {len(items):3d} 個 (換行: {newline_ok}/{len(items)})\n"
        
        # PUA精確分組統計
        if analysis["by_pua_group"]:
<<<<<<< HEAD
            report += f"\n🎯 PUA精確分組 (零容忍分類):\n"
=======
            report += f"\n🎯 PUA精確分組 (分類):\n"
>>>>>>> master
            for group_name, group_data in analysis["by_pua_group"].items():
                format_type = group_data.get("format_type", "未知")
                level = group_data.get("level", 0)
                description = group_data.get("description", "")
                count = group_data["count"]
                report += f"  L{level} {format_type:15s}: {count:3d} 個 - {description}\n"
                
                # 顯示該組的前5個符號作為示例
                symbols_preview = group_data["symbols"][:5]
                for symbol_info in symbols_preview:
                    report += f"    └─ 行{symbol_info['line']:4d}: {symbol_info['symbol']} ({symbol_info['unicode']})\n"
                if len(group_data["symbols"]) > 5:
                    report += f"    └─ ... 還有 {len(group_data['symbols']) - 5} 個\n"
        
        # 按格式類型統計
        if analysis["by_format_type"]:
            report += f"\n📊 按格式類型分布:\n"
            for format_type, count in sorted(analysis["by_format_type"].items()):
                report += f"  {format_type:15s}: {count:3d} 個\n"
        
        if analysis["by_level"]:
            report += f"\n📊 按層級分布:\n"
            for level, count in sorted(analysis["by_level"].items()):
                report += f"  Level {level}: {count:3d} 個\n"
        
        report += f"\n🔤 Unicode 字符統計 (前20個):\n"
        sorted_unicode = sorted(analysis["unicode_distribution"].items(), 
                              key=lambda x: x[1], reverse=True)
        for unicode_code, count in sorted_unicode[:20]:
            # 找到對應的字符
            char = chr(int(unicode_code[2:], 16))
            pua_mark = " [PUA]" if any(marker.unicode_code == unicode_code and marker.is_pua 
                                     for marker in markers) else ""
            report += f"  {unicode_code}: '{char}' ({count} 次){pua_mark}\n"
        
<<<<<<< HEAD
        report += f"\n📋 終極嚴格結構預覽 (前15個):\n"
=======
        report += f"\n📋 嚴格結構預覽 (前15個):\n"
>>>>>>> master
        for i, item in enumerate(analysis["structure"][:15], 1):
            level_str = f"L{item['level']}"
            format_type = f"[{item['format_type']}]" if item['format_type'] else ""
            indent = "  " * (item['level'] if isinstance(item['level'], int) else 0)
            newline_status = "✓" if item["newline_ok"] else "✗"
            pua_status = "[PUA]" if item["is_pua"] else ""
            report += f"{indent}{level_str} 行{item['line']:4d} {newline_status}: {item['symbol']} ({item['unicode']}) {pua_status}{format_type} - {item['content'][:50]}...\n"
        
        if len(analysis["structure"]) > 15:
            report += f"  ... 還有 {len(analysis['structure']) - 15} 個標記\n"
        
<<<<<<< HEAD
        report += f"\n⚡ 終極嚴格性分析:\n"
        report += f"  檢測精度: 100% (符合終極嚴格規則)\n"
        report += f"  零妥協率: 100% (無例外處理)\n"
=======
        report += f"\n⚡ 嚴格性分析:\n"
>>>>>>> master
        report += f"  字符範圍: {len(self.all_valid_chars)} 個預定義 + {len(self.pua_symbol_groups)} 個PUA分組\n"
        report += f"  PUA精確度: {len([g for g in analysis['by_pua_group']]) if 'by_pua_group' in analysis else 0} 個精確分組\n"
        
        return report

<<<<<<< HEAD
def main():
    INPUT_FILE = "data/sample/TPDM,111,易,564,20250113,1.json"
    
    print("⚡ 啟動終極嚴格格式層級檢測器")
    print("規則：\\r\\n + 單字符 + 、(零例外)")
    print()
=======
    def process_artifact(self, artifact: JudgmentArtifact) -> None:
        """接收 artifact 並處理，直接標記在 DocumentLine 的 detection 屬性上"""
        ultra_strict_pattern = r'^(.{1})、(.*)$'
        markers_count = 0
        
        for doc_line in artifact.full_lines:
            stripped_line = doc_line.original_text.strip()
            
            if not stripped_line:
                doc_line.detection = SymbolDetection(method_used="empty_line")
                continue
                
            match = re.match(ultra_strict_pattern, stripped_line)
            if match:
                symbol = match.group(1)
                
                is_valid, category = self.is_valid_symbol(symbol)
                if is_valid:
                    has_proper_newline = True
                    if doc_line.index > 0:
                        prev_line = artifact.full_lines[doc_line.index - 1].original_text
                        has_proper_newline = prev_line.endswith('\r')
                    
                    if has_proper_newline:
                        doc_line.detection = SymbolDetection(
                            detected_symbol=symbol,
                            symbol_category=category,
                            is_pua=self.is_pua_character(symbol),
                            method_used="ultra_strict_pua",
                            rule_based_score=1.0,
                            bert_score=1.0,
                            assigned_level=self.get_symbol_level(symbol, category)
                        )
                        markers_count += 1
                        
        artifact.processing_stats['ultra_strict_markers_count'] = markers_count
    
def main():
    INPUT_FILE = "data/samples/TPDM,111,易,564,20250113,1.json"
    
    print("⚡ 啟動嚴格格式層級檢測器")
>>>>>>> master
    
    try:
        # 讀取JSON文件
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'JFULL' not in data:
            raise ValueError("JSON檔案中找不到 'JFULL' 欄位")
        
        text_lines = data['JFULL'].split('\n')  # 保留原始換行符
        print(f"📄 讀取文本：{len(text_lines)} 行")
        
<<<<<<< HEAD
        # 終極嚴格檢測
        detector = UltraStrictDetector()
        markers = detector.detect_ultra_strict_markers(text_lines)
        
        print(f"✅ 終極嚴格檢測完成：發現 {len(markers)} 個符合格式的標記")
=======
        # 嚴格檢測
        detector = UltraStrictDetector()
        markers = detector.detect_ultra_strict_markers(text_lines)
        
        print(f"✅ 嚴格檢測完成：發現 {len(markers)} 個符合格式的標記")
>>>>>>> master
        
        # 分析結構
        analysis = detector.analyze_ultra_strict_structure(markers)
        
        # 生成報告
        report = detector.generate_ultra_strict_report(markers, analysis)
        print(report)
        
        # 保存結果
        output_data = {
            "input_file": INPUT_FILE,
            "detection_method": "ultra_strict_format",
            "ultra_strict_rules": [
                "必須以 \\r\\n 開頭",
                "只能是單個 Unicode 字符",
                "必須緊跟 、(頓號)",
<<<<<<< HEAD
                "字符必須在預定義範圍內（含PUA）",
                "零例外，零妥協"
=======
                "字符必須在預定義範圍內（含PUA）"
>>>>>>> master
            ],
            "total_markers": len(markers),
            "pua_markers": analysis["pua_count"],
            "newline_compliance": analysis["newline_compliance"],
            "markers": [
                {
                    "line": m.line_number,
                    "symbol": m.symbol,
                    "unicode": m.unicode_code,
                    "category": m.category,
                    "is_pua": m.is_pua,
                    "has_proper_newline": m.has_proper_newline,
                    "content": m.content
                } for m in markers
            ],
            "analysis": analysis
        }
        
        output_file = "ultra_strict_detection.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 詳細結果已保存: {output_file}")
        
        # 提供字符範圍資訊
        print(f"\n🔤 支援的字符範圍:")
        for category, info in detector.valid_symbol_ranges.items():
            print(f"  {category:15s}: {len(info['chars'])} 個字符")
        print(f"  PUA字符範圍: {len(detector.pua_ranges)} 個範圍")
        
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 {INPUT_FILE}")
    except json.JSONDecodeError:
        print("❌ 錯誤：JSON 格式不正確")
    except Exception as e:
        print(f"❌ 執行錯誤：{e}")

if __name__ == "__main__":
    main()