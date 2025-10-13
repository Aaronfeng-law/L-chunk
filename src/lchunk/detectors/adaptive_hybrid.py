#!/usr/bin/env python3
"""
自適應混合層級符號檢測器 (Intelligent Hybrid Detector)
先學習再應用" 原則：文件分塊 → 規則學習 → 全文應用

處理流程：
1. 文件分塊：使用 comprehensive_analysis 分析文件結構
2. 全文層級符號偵測：用 hybrid_detector 檢測所有符號
3. 規則學習區間：在 R-D 或 S-D 區間建立層級規則
4. 層級規則建立：分析符號類型和層級模式
5. 全文應用：將學習到的規則應用到整個文件

"Good code teaches. Great code learns and then teaches." -
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
# warnings.filterwarnings('ignore')

# 導入現有模組
sys.path.append('.')
from .hybrid import HybridLevelSymbolDetector, HybridDetectionResult
from ..analyzers.splitter import process_single_file, find_section_patterns
from ..analyzers.comprehensive import analyze_filtered_dataset

@dataclass
class LevelingRule:
    """層級規則定義"""
    symbol_category: str
    assigned_level: int
    confidence: float
    learning_source: str  # "R-D", "S-D", "全文"
    occurrences: int
    examples: List[str]

@dataclass
class IntelligentDetectionResult:
    """自適應檢測結果"""
    filename: str
    file_structure: Dict  # comprehensive_analysis 結果
    learning_region: str  # "R-D", "S-D", "全文"
    learned_rules: List[LevelingRule]
    full_detection_results: List[HybridDetectionResult]
    applied_hierarchy: Dict
    processing_stats: Dict

class IntelligentHybridDetector:
    """自適應混合層級符號檢測器"""
    
    def __init__(self, model_path: Optional[str] = None):
        # 初始化基礎檢測器 - 只在有模型時才載入 BERT
        self.hybrid_detector = HybridLevelSymbolDetector(model_path if model_path else None)
        
        # 自適應檢測結果
        self.detection_results = []
        
        print("🧠 自適應混合檢測器已初始化")
        print("策略：文件分塊 → 規則學習 → 全文應用")
    
    def analyze_file_structure(self, file_path: Path) -> Tuple[bool, Dict]:
        """分析檔案結構 - 使用 comprehensive_analysis 的邏輯"""
        try:
            # 使用 judgment_splitter 處理單個檔案
            success, result = process_single_file(file_path)
            
            if not success:
                return False, {}
            
            # 讀取原始數據
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 分析章節結構
            sections = result['sections']
            has_main_text = bool(sections.get('main_text', []))
            has_facts = bool(sections.get('facts', []))
            has_reasons = bool(sections.get('reasons', []))
            has_facts_and_reasons = bool(sections.get('facts_and_reasons', []))
            
            # 確定學習區間類型
            learning_region = None
            learning_lines = []
            
            if has_facts_and_reasons:
                # S-D 區間：從 facts_and_reasons 到文件末尾
                learning_region = "S-D"
                fr_lines = sections.get('facts_and_reasons', [])
                if fr_lines:
                    # 獲取 facts_and_reasons 開始的行號
                    full_lines = data['JFULL'].split('\n')
                    fr_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [l.strip() for l in fr_lines[:3]]:
                            fr_start_line = i
                            break
                    
                    if fr_start_line is not None:
                        learning_lines = full_lines[fr_start_line:]
            
            elif has_reasons:
                # R-D 區間：從 reasons 到文件末尾
                learning_region = "R-D"
                reasons_lines = sections.get('reasons', [])
                if reasons_lines:
                    full_lines = data['JFULL'].split('\n')
                    reasons_start_line = None
                    for i, line in enumerate(full_lines):
                        if line.strip() and line.strip() in [l.strip() for l in reasons_lines[:3]]:
                            reasons_start_line = i
                            break
                    
                    if reasons_start_line is not None:
                        learning_lines = full_lines[reasons_start_line:]
            
            if not learning_region:
                # 沒有 R 或 S 章節，使用全文
                learning_region = "全文"
                learning_lines = data['JFULL'].split('\n')
            
            structure_info = {
                'sections': sections,
                'has_main_text': has_main_text,
                'has_facts': has_facts,
                'has_reasons': has_reasons,
                'has_facts_and_reasons': has_facts_and_reasons,
                'learning_region': learning_region,
                'learning_lines': learning_lines,
                'full_text_lines': data['JFULL'].split('\n'),
                'total_lines': len(data['JFULL'].split('\n'))
            }
            
            return True, structure_info
        
        except Exception as e:
            print(f"❌ 分析檔案結構失敗: {e}")
            return False, {}
    
    def learn_leveling_rules(self, learning_lines: List[str], learning_region: str) -> List[LevelingRule]:
        """在學習區間建立層級規則 - 完全動態學習
        
        不再依賴任何預定義層級，完全基於文件本身的符號出現順序
        """
        print(f"🎓 在 {learning_region} 區間學習層級規則...")
        print(f"   學習範圍: {len(learning_lines)} 行")
        
        # 在學習區間執行檢測
        learning_results = self.hybrid_detector.detect_hybrid_markers(learning_lines)
        
        # 獲取學習區間的層級分析
        self.hybrid_detector.detection_results = learning_results
        hierarchy_analysis = self.hybrid_detector.detect_hierarchy_levels()
        
        if not hierarchy_analysis or not hierarchy_analysis.get('level_mapping'):
            print("⚠️ 學習區間未發現有效的層級規則")
            return []
        
        # 建立規則 - 完全基於學習的層級
        rules = []
        level_mapping = hierarchy_analysis['level_mapping']
        
        print(f"✅ 學習到 {len(level_mapping)} 種符號類型的層級規則")
        
        for symbol_category, level_info in level_mapping.items():
            rule = LevelingRule(
                symbol_category=symbol_category,
                assigned_level=level_info['assigned_level'],
                confidence=level_info['count'] / len([r for r in learning_results if r.final_prediction]),
                learning_source=learning_region,
                occurrences=level_info['count'],
                examples=[ex['text'][:50] + '...' for ex in level_info['examples'][:3]]
            )
            rules.append(rule)
            
            print(f"   📋 {symbol_category}: Level {rule.assigned_level} (信心度: {rule.confidence:.3f})")
        
        return rules
    
    def apply_leveling_rules(self, full_results: List[HybridDetectionResult], 
                           learned_rules: List[LevelingRule]) -> Dict:
        """將學習到的規則應用到全文檢測結果"""
        print("🔧 將學習規則應用到全文...")
        
        # 建立規則映射
        rule_mapping = {}
        for rule in learned_rules:
            rule_mapping[rule.symbol_category] = rule.assigned_level
        
        # 應用規則到全文結果
        enhanced_hierarchy = []
        unknown_categories = set()
        next_available_level = max(rule_mapping.values()) + 1 if rule_mapping else 1
        
        for result in full_results:
            if not result.final_prediction:
                continue
            
            symbol_category = result.symbol_category
            
            if symbol_category in rule_mapping:
                # 使用學習到的規則
                assigned_level = rule_mapping[symbol_category]
            else:
                # 新的符號類型，分配新層級
                if symbol_category not in unknown_categories:
                    rule_mapping[symbol_category] = next_available_level
                    unknown_categories.add(symbol_category)
                    next_available_level += 1
                
                assigned_level = rule_mapping[symbol_category]
            
            enhanced_hierarchy.append({
                'line_number': result.line_number,
                'detected_symbol': result.detected_symbol,
                'symbol_category': symbol_category,
                'assigned_level': assigned_level,
                'is_learned_rule': symbol_category not in unknown_categories,
                'line_text': result.line_text,
                'method_used': result.method_used,
                'bert_score': result.bert_score
            })
        
        # 創建層級映射統計
        level_stats = {}
        for item in enhanced_hierarchy:
            category = item['symbol_category']
            if category not in level_stats:
                level_stats[category] = {
                    'assigned_level': item['assigned_level'],
                    'count': 0,
                    'is_learned': item['is_learned_rule'],
                    'examples': []
                }
            level_stats[category]['count'] += 1
            if len(level_stats[category]['examples']) < 3:
                level_stats[category]['examples'].append({
                    'line': item['line_number'],
                    'symbol': item['detected_symbol'],
                    'text': item['line_text'][:50] + '...'
                })
        
        print(f"✅ 規則應用完成:")
        print(f"   已知規則: {len(rule_mapping) - len(unknown_categories)} 種")
        print(f"   新發現: {len(unknown_categories)} 種")
        print(f"   總層級符號: {len(enhanced_hierarchy)} 個")
        
        return {
            'enhanced_hierarchy': enhanced_hierarchy,
            'level_mapping': level_stats,
            'rule_coverage': (len(rule_mapping) - len(unknown_categories)) / len(rule_mapping) if rule_mapping else 0,
            'total_levels': len(set(item['assigned_level'] for item in enhanced_hierarchy)),
            'total_symbols': len(enhanced_hierarchy)
        }
    
    def process_single_file(self, file_path: Path) -> Optional[IntelligentDetectionResult]:
        """處理單個檔案 - 完整的自適應檢測流程"""
        print(f"\n🔍 自適應檢測: {file_path.name}")
        
        # 步驟1: 文件分塊
        success, structure_info = self.analyze_file_structure(file_path)
        if not success:
            print(f"❌ 檔案結構分析失敗")
            return None
        
        learning_region = structure_info['learning_region']
        print(f"📊 檔案結構: {learning_region} 模式")
        
        # 步驟2: 全文層級符號偵測
        print("🔎 執行全文層級符號檢測...")
        full_text_lines = structure_info['full_text_lines']
        full_detection_results = self.hybrid_detector.detect_hybrid_markers(full_text_lines)
        
        # 步驟3: 規則學習區間
        learning_lines = structure_info['learning_lines']
        learned_rules = self.learn_leveling_rules(learning_lines, learning_region)
        
        # 步驟4: 層級規則建立與全文應用
        applied_hierarchy = self.apply_leveling_rules(full_detection_results, learned_rules)
        
        # 處理統計
        processing_stats = {
            'total_lines': structure_info['total_lines'],
            'learning_lines': len(learning_lines),
            'total_symbols_detected': len([r for r in full_detection_results if r.final_prediction]),
            'learned_rules_count': len(learned_rules),
            'rule_coverage': applied_hierarchy['rule_coverage'],
            'final_levels': applied_hierarchy['total_levels']
        }
        
        result = IntelligentDetectionResult(
            filename=file_path.name,
            file_structure=structure_info,
            learning_region=learning_region,
            learned_rules=learned_rules,
            full_detection_results=full_detection_results,
            applied_hierarchy=applied_hierarchy,
            processing_stats=processing_stats
        )
        
        return result
    
    def process_sample_directory(self, sample_dir: Path):
        """處理 sample 目錄中的所有檔案"""
        print(f"🚀 自適應批量檢測: {sample_dir}")
        print("="*80)
        
        if not sample_dir.exists():
            print(f"❌ 目錄不存在: {sample_dir}")
            return
        
        json_files = list(sample_dir.glob("*.json"))
        if not json_files:
            print(f"❌ 在 {sample_dir} 中沒有找到 JSON 檔案")
            return
        
        print(f"📁 找到 {len(json_files)} 個檔案")
        
        all_results = []
        learning_region_stats = {'S-D': 0, 'R-D': 0, '全文': 0}
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] 處理: {json_file.name}")
            
            result = self.process_single_file(json_file)
            if result:
                all_results.append(result)
                learning_region_stats[result.learning_region] += 1
            else:
                print(f"❌ 處理失敗: {json_file.name}")
        
        # 生成綜合報告
        self.generate_batch_report(all_results, learning_region_stats)
    
    def generate_batch_report(self, results: List[IntelligentDetectionResult], 
                            region_stats: Dict):
        """生成批量處理報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        report_file = output_dir / f"adaptive_detection_report_{timestamp}.md"
        
        report = f"""# 自適應混合層級符號檢測報告
生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
處理檔案: {len(results)} 個

## 📊 整體統計

### 學習區間分布
- **S-D 區間** (事實理由合併): {region_stats['S-D']} 檔案
- **R-D 區間** (理由章節): {region_stats['R-D']} 檔案  
- **全文檢測**: {region_stats['全文']} 檔案

### 處理統計
"""
        
        if results:
            total_lines = sum(r.processing_stats['total_lines'] for r in results)
            total_symbols = sum(r.processing_stats['total_symbols_detected'] for r in results)
            avg_coverage = sum(r.processing_stats['rule_coverage'] for r in results) / len(results)
            
            report += f"- 總行數: {total_lines:,}\n"
            report += f"- 總符號數: {total_symbols:,}\n"
            report += f"- 平均規則覆蓋率: {avg_coverage:.1%}\n"
            
            # 各檔案詳細結果
            report += "\n## 📋 各檔案檢測結果\n\n"
            
            for i, result in enumerate(results, 1):
                stats = result.processing_stats
                hierarchy = result.applied_hierarchy
                
                report += f"### {i}. {result.filename}\n"
                report += f"- **學習模式**: {result.learning_region}\n"
                report += f"- **總行數**: {stats['total_lines']:,}\n"
                report += f"- **學習範圍**: {stats['learning_lines']:,} 行\n"
                report += f"- **檢測符號**: {stats['total_symbols_detected']:,} 個\n"
                report += f"- **學習規則**: {stats['learned_rules_count']} 種\n"
                report += f"- **規則覆蓋**: {stats['rule_coverage']:.1%}\n"
                report += f"- **最終層級**: {stats['final_levels']} 層\n"
                
                # 顯示學習到的規則
                if result.learned_rules:
                    report += f"\n**學習規則:**\n"
                    for rule in result.learned_rules[:5]:  # 只顯示前5個
                        report += f"  - {rule.symbol_category}: L{rule.assigned_level} (信心度: {rule.confidence:.3f})\n"
                
                # 顯示層級結構預覽
                if hierarchy.get('enhanced_hierarchy'):
                    report += f"\n**層級結構預覽:**\n"
                    for item in hierarchy['enhanced_hierarchy'][:5]:  # 只顯示前5個
                        learned_mark = "✓" if item['is_learned_rule'] else "✗"
                        indent = "  " * item['assigned_level']
                        report += f"  {indent}L{item['assigned_level']} {learned_mark} 行{item['line_number']:4}: {item['detected_symbol']} - {item['line_text'][:40]}...\n"
                
                report += "\n"
        
        # 保存報告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 自適應檢測報告已保存: {report_file}")
        
        # 保存詳細數據
        json_file = output_dir / f"adaptive_detection_data_{timestamp}.json"
        json_data = []
        
        for result in results:
            # 轉換為可序列化的格式
            json_data.append({
                'filename': result.filename,
                'learning_region': result.learning_region,
                'processing_stats': result.processing_stats,
                'learned_rules': [
                    {
                        'symbol_category': rule.symbol_category,
                        'assigned_level': rule.assigned_level,
                        'confidence': rule.confidence,
                        'learning_source': rule.learning_source,
                        'occurrences': rule.occurrences,
                        'examples': rule.examples
                    } for rule in result.learned_rules
                ],
                'applied_hierarchy': result.applied_hierarchy
            })
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"📊 詳細數據已保存: {json_file}")

def main():
    """主函數 - 自適應檢測演示"""
    print("🧠 自適應混合層級符號檢測器")
    print("基於  '先學習再應用' 原則")
    print("文件分塊 → 規則學習 → 全文應用")
    print("="*80)
    
    # 初始化自適應檢測器
    model_path = "models/bert/level_detector/best_model"
    detector = IntelligentHybridDetector(model_path if Path(model_path).exists() else None)

    # 處理 filtered 目錄
    filtered_dir = Path("data/processed/filtered")
    detector.process_sample_directory(filtered_dir)

if __name__ == "__main__":
    main()
