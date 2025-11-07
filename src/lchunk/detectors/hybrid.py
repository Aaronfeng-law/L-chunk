#!/usr/bin/env python3
"""
三層混合層級符號檢測器 (推理專用)
"分層過濾" 原則：嚴格 → 軟規則 → 聚合

三層策略：
1. 終極嚴格規則：PUA字符 + 頓號 = 100% 確定
2. 軟規則 + BERT：其他符號需要語義驗證
3. 最終聚合：合併所有檢測結果

注意：此版本只負責推理，不包含訓練功能
訓練請使用 train_bert_classifier.py
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入現有檢測器
from .ultra_strict import UltraStrictDetector

# BERT相關
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np

def convert_numpy_types(obj):
    """轉換 numpy 類型為 Python 原生類型 - """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

@dataclass
class HybridDetectionResult:
    """混合檢測結果"""
    line_number: int
    line_text: str
    detected_symbol: str
    symbol_category: str
    rule_based_score: float  # 規則檢測信心度 (0, 0.5, 1.0)
    bert_score: float        # BERT分類信心度 (0-1)
    final_prediction: bool   # 最終預測結果
    method_used: str         # 使用的方法 ("ultra_strict_pua", "soft_rule_bert", "rule_rejected", etc.)

class HybridLevelSymbolDetector:
    """三層混合層級符號檢測器 (推理專用)"""
    
    def __init__(self, model_path: Optional[str] = None):
        # 初始化規則檢測器
        self.rule_detector = UltraStrictDetector()
        
        # BERT 模型相關
        self.bert_model = None
        self.bert_tokenizer = None
        self.model_path = model_path
        
        # 檢測結果
        self.detection_results = []
        
        # 如果提供了模型路徑且路徑存在，直接載入
        if model_path is not None and Path(model_path).exists():
            self.load_bert_model(model_path)
    
    def load_bert_model(self, model_path: Optional[str] = None):
        """載入訓練好的 BERT 模型"""
        if model_path:
            self.model_path = Path(model_path)
        elif self.model_path:
            self.model_path = Path(self.model_path)
        else:
            raise ValueError("請提供模型路徑")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路徑不存在: {self.model_path}")
        
        print(f"📦 載入 BERT 模型: {self.model_path}")
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # 自適應設備選擇 - 優先使用 GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = self.bert_model.to(device)
        self.bert_model.eval()
        
        print(f"✅ BERT 模型載入完成 (device: {device})")
    
    def is_model_loaded(self) -> bool:
        """檢查 BERT 模型是否已載入"""
        return self.bert_model is not None and self.bert_tokenizer is not None
    
    def _is_ultra_strict_pua_symbol(self, line_text: str) -> Tuple[bool, str, str]:
        """檢查是否為終極嚴格的 PUA 符號：PUA字符 + 頓號"""
        if not line_text or len(line_text.strip()) < 2:
            return False, None, None
        
        clean_text = line_text.strip()
        first_char = clean_text[0]
        
        # 檢查是否為 PUA 字符
        if not self.rule_detector.is_pua_character(first_char):
            return False, None, None
        
        # 檢查是否緊跟頓號
        if len(clean_text) > 1 and clean_text[1] == '、':
            # 確定 PUA 分組
            pua_group = self.rule_detector.get_pua_group(first_char)
            return True, first_char, pua_group or "PUA_未分類"
        
        return False, None, None
    
    def _get_line_symbol_info(self, line_text: str) -> Tuple[bool, str, str]:
        """檢查行是否以層級符號開頭（軟規則）"""
        if not line_text or len(line_text.strip()) == 0:
            return False, None, None
        
        first_char = line_text.strip()[0]
        
        # 檢查是否在預定義的符號範圍內
        for category, info in self.rule_detector.valid_symbol_ranges.items():
            if first_char in info["chars"]:
                return True, first_char, category
        
        # 檢查PUA符號（但不要求頓號）
        for pua_category, pua_info in self.rule_detector.pua_symbol_groups.items():
            if 'chars' in pua_info and first_char in pua_info['chars']:
                return True, first_char, pua_category
        
        return False, None, None
    
    def bert_classify_lines(self, lines: List[str], batch_size: int = 32) -> List[Tuple[float, int]]:
        """使用 BERT 對行進行分類 - 優化 GPU 推理性能，支持分批處理避免 OOM"""
        if not self.is_model_loaded():
            raise ValueError("請先載入 BERT 模型")
        
        if not lines:
            return []
        
        try:
            # 自適應設備選擇
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model = self.bert_model.to(device)
            
            all_results = []
            
            # 分批處理以避免 OOM
            for batch_start in range(0, len(lines), batch_size):
                batch_end = min(batch_start + batch_size, len(lines))
                batch_lines = lines[batch_start:batch_end]
                
                # 準備輸入
                inputs = self.bert_tokenizer(
                    batch_lines,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # 將輸入移到同一設備
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 預測
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    scores = probabilities[:, 1]  # 正類別的概率
                
                # 返回結果到 CPU 並保存
                batch_results = list(zip(scores.cpu().numpy(), predictions.cpu().numpy()))
                all_results.extend(batch_results)
                
                # 清理中間張量
                del inputs, outputs, probabilities, predictions, scores
                
                # 在每個批次後清理 CUDA 快取
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return all_results
        
        finally:
            # 最終清理 CUDA 快取
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def detect_hybrid_markers(self, text_lines: List[str], bert_threshold: float = 0.5, verbose: bool = True) -> List[HybridDetectionResult]:
        """三層混合檢測層級標記 - """
        if verbose:
            print("🔍 啟動三層混合檢測...")
        
        results = []
        ultra_strict_results = []
        soft_candidate_lines = []
        soft_line_mapping = {}
        
        # 第一層：終極嚴格規則 (PUA + 頓號) - 100% 確定
        if verbose:
            print("🎯 步驟1: 終極嚴格規則 (PUA + 頓號)")
        
        for line_num, line_text in enumerate(text_lines, 1):
            clean_text = line_text.strip()
            if not clean_text:
                # 空行記錄為負類
                results.append(HybridDetectionResult(
                    line_number=line_num,
                    line_text=clean_text,
                    detected_symbol=None,
                    symbol_category=None,
                    rule_based_score=0.0,
                    bert_score=0.0,
                    final_prediction=False,
                    method_used="empty_line"
                ))
                continue
            
            # 檢查終極嚴格規則
            is_ultra_strict, detected_symbol, symbol_category = self._is_ultra_strict_pua_symbol(clean_text)
            
            if is_ultra_strict:
                # PUA + 頓號 = 100% 確定的層級符號
                ultra_strict_results.append(HybridDetectionResult(
                    line_number=line_num,
                    line_text=clean_text,
                    detected_symbol=detected_symbol,
                    symbol_category=symbol_category,
                    rule_based_score=1.0,
                    bert_score=1.0,  # 100% 信心
                    final_prediction=True,
                    method_used="ultra_strict_pua"
                ))
            else:
                # 檢查軟規則 (其他符號開頭)
                is_symbol_line, soft_symbol, soft_category = self._get_line_symbol_info(clean_text)
                
                if is_symbol_line:
                    # 軟規則候選，需要 BERT 驗證
                    soft_candidate_lines.append(clean_text)
                    soft_line_mapping[len(soft_candidate_lines) - 1] = {
                        'line_number': line_num,
                        'line_text': clean_text,
                        'detected_symbol': soft_symbol,
                        'symbol_category': soft_category
                    }
                else:
                    # 非候選行記錄為負類
                    results.append(HybridDetectionResult(
                        line_number=line_num,
                        line_text=clean_text,
                        detected_symbol=None,
                        symbol_category=None,
                        rule_based_score=0.0,
                        bert_score=0.0,
                        final_prediction=False,
                        method_used="rule_rejected"
                    ))
        
        if verbose:
            print(f"✅ 終極嚴格規則確定 {len(ultra_strict_results)} 個層級符號")
            print(f"📋 軟規則找到 {len(soft_candidate_lines)} 個候選行")
        
        # 第二層：軟規則 + BERT 分類
        if soft_candidate_lines and self.is_model_loaded():
            if verbose:
                print("🤖 步驟2: BERT 精細分類軟規則候選...")
            bert_results = self.bert_classify_lines(soft_candidate_lines)
            
            for i, (bert_score, bert_prediction) in enumerate(bert_results):
                line_info = soft_line_mapping[i]
                
                # 基於 BERT 閾值決定最終結果
                final_prediction = bert_score >= bert_threshold
                method_used = "soft_rule_bert"
                
                results.append(HybridDetectionResult(
                    line_number=line_info['line_number'],
                    line_text=line_info['line_text'],
                    detected_symbol=line_info['detected_symbol'],
                    symbol_category=line_info['symbol_category'],
                    rule_based_score=0.5,  # 軟規則通過
                    bert_score=float(bert_score),
                    final_prediction=final_prediction,
                    method_used=method_used
                ))
        
        elif soft_candidate_lines:
            # 沒有 BERT 模型，軟規則候選全部接受
            if verbose:
                print("⚠️ 沒有 BERT 模型，軟規則候選全部接受")
            for i, line_info in soft_line_mapping.items():
                results.append(HybridDetectionResult(
                    line_number=line_info['line_number'],
                    line_text=line_info['line_text'],
                    detected_symbol=line_info['detected_symbol'],
                    symbol_category=line_info['symbol_category'],
                    rule_based_score=0.5,
                    bert_score=0.0,
                    final_prediction=True,  # 軟規則為準
                    method_used="soft_rule_only"
                ))
        
        # 第三層：最終聚合
        if verbose:
            print("📊 步驟3: 聚合所有檢測結果...")
        
        # 合併終極嚴格和軟規則結果
        all_results = ultra_strict_results + results
        
        # 按行號排序
        all_results.sort(key=lambda x: x.line_number)
        self.detection_results = all_results
        
        # 統計結果
        ultra_strict_count = len(ultra_strict_results)
        soft_rule_accepted = sum(1 for r in results if r.final_prediction and r.method_used.startswith("soft_rule"))
        total_detected = ultra_strict_count + soft_rule_accepted
        
        if verbose:
            print(f"✅ 三層檢測完成，處理了 {len(text_lines)} 行")
            print(f"   🎯 終極嚴格: {ultra_strict_count} 個 (100% 確定)")
            print(f"   🤖 軟規則+BERT: {soft_rule_accepted} 個")
            print(f"   📊 總檢測: {total_detected} 個層級符號")
        
        return all_results
    
    def detect_hierarchy_levels(self) -> Dict:
        """自動檢測層級結構 - 後往前分析
        
        基於 ultra_strict_detector 的符號群組層級定義
        從檔案末尾開始檢測，追蹤新符號類型出現，分配遞增層級
        """
        if not self.detection_results:
            return {}
        
        # 獲取所有正類結果（實際檢測到的層級符號）
        positive_results = [r for r in self.detection_results if r.final_prediction]
        
        if not positive_results:
            return {'hierarchy_levels': [], 'level_mapping': {}}
        
        # 依行號順序遍歷，遇到新符號類型時建立下一層
        hierarchy_levels = []
        category_levels: Dict[str, int] = {}
        current_level = 1

        for result in positive_results:
            symbol_category = result.symbol_category

            # 移除預定義層級，完全動態學習
            predefined_level = 0

            if symbol_category not in category_levels:
                category_levels[symbol_category] = current_level
                current_level += 1

            assigned_level = category_levels[symbol_category]

            hierarchy_levels.append({
                'line_number': result.line_number,
                'detected_symbol': result.detected_symbol,
                'symbol_category': symbol_category,
                'predefined_level': predefined_level,
                'assigned_level': assigned_level,
                'is_pua': self.rule_detector.is_pua_character(result.detected_symbol) if result.detected_symbol else False,
                'line_text': result.line_text,
                'method_used': result.method_used,
                'bert_score': result.bert_score
            })

        # 創建層級映射表
        level_mapping = {}
        for item in hierarchy_levels:
            category = item['symbol_category']
            if category not in level_mapping:
                level_mapping[category] = {
                    'assigned_level': item['assigned_level'],
                    'count': 0,
                    'is_pua_category': item['is_pua'],
                    'examples': []
                }
            level_mapping[category]['count'] += 1
            if len(level_mapping[category]['examples']) < 3:
                level_mapping[category]['examples'].append({
                    'line': item['line_number'],
                    'symbol': item['detected_symbol'],
                    'text': item['line_text'][:50] + '...' if len(item['line_text']) > 50 else item['line_text']
                })
        
        return {
            'hierarchy_levels': hierarchy_levels,
            'level_mapping': level_mapping,
            'total_levels': len(category_levels),
            'total_symbols': len(hierarchy_levels)
        }
    
    def analyze_detection_results(self) -> Dict:
        """分析檢測結果 - 包含層級結構分析"""
        if not self.detection_results:
            return {}
        
        total_lines = len(self.detection_results)
        positive_predictions = sum(1 for r in self.detection_results if r.final_prediction)
        
        # 按方法統計
        method_stats = {}
        for result in self.detection_results:
            method = result.method_used
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'positive': 0}
            method_stats[method]['total'] += 1
            if result.final_prediction:
                method_stats[method]['positive'] += 1
        
        # 符號類別統計
        symbol_stats = {}
        for result in self.detection_results:
            if result.final_prediction and result.symbol_category:
                category = result.symbol_category
                if category not in symbol_stats:
                    symbol_stats[category] = 0
                symbol_stats[category] += 1
        
        # BERT 信心度分析
        bert_scores = [r.bert_score for r in self.detection_results if r.bert_score > 0 and r.method_used.startswith("soft_rule")]
        
        # PUA 終極嚴格統計
        ultra_strict_count = sum(1 for r in self.detection_results if r.method_used == "ultra_strict_pua")
        
        # 自動層級檢測
        hierarchy_analysis = self.detect_hierarchy_levels()
        
        analysis = {
            'total_lines': total_lines,
            'positive_predictions': positive_predictions,
            'positive_ratio': positive_predictions / total_lines if total_lines > 0 else 0,
            'ultra_strict_count': ultra_strict_count,
            'method_statistics': method_stats,
            'symbol_category_distribution': symbol_stats,
            'hierarchy_analysis': hierarchy_analysis,
            'bert_score_stats': {
                'mean': np.mean(bert_scores) if bert_scores else 0,
                'std': np.std(bert_scores) if bert_scores else 0,
                'min': np.min(bert_scores) if bert_scores else 0,
                'max': np.max(bert_scores) if bert_scores else 0,
                'count': len(bert_scores)
            }
        }
        
        return analysis
    
    def generate_detection_report(self, analysis: Dict) -> str:
        """生成檢測報告 - 包含層級結構分析"""
        report = f"""
============================================================
🔬 三層混合層級符號檢測報告 (層過濾)
============================================================

📊 總體統計:
  處理行數: {analysis['total_lines']:,}
  檢測到層級符號: {analysis['positive_predictions']:,} ({analysis['positive_ratio']:.1%})

📈 按檢測方法統計:
"""
        
        for method, stats in analysis['method_statistics'].items():
            positive_rate = stats['positive'] / stats['total'] if stats['total'] > 0 else 0
            method_name = {
                'ultra_strict_pua': '終極嚴格(PUA+頓號)',
                'soft_rule_bert': 'BERT精煉',
                'soft_rule_only': '軟規則檢測',
                'rule_rejected': '規則拒絕',
                'empty_line': '空行'
            }.get(method, method)
            report += f"  {method_name:15} : {stats['total']:4} 行, {stats['positive']:4} 正類 ({positive_rate:.1%})\n"
        
        if analysis['symbol_category_distribution']:
            report += f"\n🎯 符號類別分布:\n"
            sorted_symbols = sorted(analysis['symbol_category_distribution'].items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_symbols:
                report += f"  {category:20} : {count:4} 個\n"
        
        # 層級結構分析報告
        hierarchy = analysis.get('hierarchy_analysis', {})
        if hierarchy and hierarchy.get('hierarchy_levels'):
            report += f"\n🏗️ 自動層級結構分析 (後往前檢測):\n"
            report += f"  檢測到層級數: {hierarchy['total_levels']} 個不同類型\n"
            report += f"  層級符號總數: {hierarchy['total_symbols']} 個\n"
            
            # 層級映射表
            if hierarchy.get('level_mapping'):
                report += f"\n📋 層級映射表 (基於 ultra_strict_detector 群組):\n"
                sorted_levels = sorted(hierarchy['level_mapping'].items(), 
                                     key=lambda x: x[1]['assigned_level'])
                
                for category, level_info in sorted_levels:
                    assigned_level = level_info['assigned_level']
                    count = level_info['count']
                    is_pua = " [PUA]" if level_info['is_pua_category'] else ""
                    
                    report += f"  Level {assigned_level}: {category:25}{is_pua} ({count} 個)\n"
                    
                    # 顯示例子
                    for example in level_info['examples'][:2]:
                        report += f"    └─ 行{example['line']:4}: {example['symbol']} - {example['text']}\n"
            
            # 層級結構預覽
            report += f"\n🔍 層級結構預覽 (前15個符號):\n"
            for item in hierarchy['hierarchy_levels'][:15]:
                level = item['assigned_level']
                symbol = item['detected_symbol']
                line_num = item['line_number']
                category = item['symbol_category']
                method_icon = "🎯" if item['method_used'] == "ultra_strict_pua" else "🤖"
                pua_mark = "[PUA]" if item['is_pua'] else ""
                bert_info = f" (BERT: {item['bert_score']:.3f})" if item['bert_score'] > 0 and item['bert_score'] < 1.0 else ""
                
                indent = "  " * level
                report += f"{indent}L{level} {method_icon} 行{line_num:4}: {symbol} {pua_mark}{bert_info} - {item['line_text'][:60]}...\n"
            
            if len(hierarchy['hierarchy_levels']) > 15:
                report += f"  ... 還有 {len(hierarchy['hierarchy_levels']) - 15} 個層級符號\n"
        
        bert_stats = analysis['bert_score_stats']
        if bert_stats['count'] > 0:
            report += f"\n🤖 BERT 信心度統計 (軟規則部分):\n"
            report += f"  處理行數:  {bert_stats['count']}\n"
            report += f"  平均信心度: {bert_stats['mean']:.3f}\n"
            report += f"  標準差: {bert_stats['std']:.3f}\n"
            report += f"  範圍: {bert_stats['min']:.3f} ~ {bert_stats['max']:.3f}\n"
        
        report += f"\n⚡ 察:\n"
        if analysis.get('ultra_strict_count', 0) > 0:
            ultra_ratio = analysis['ultra_strict_count'] / analysis['positive_predictions'] if analysis['positive_predictions'] > 0 else 0
            report += f"  終極嚴格比例: {ultra_ratio:.1%} - PUA+頓號格式標準化程度\n"
        
        if analysis['positive_ratio'] > 0.5:
            report += f"  高密度文檔 ({analysis['positive_ratio']:.1%}) - 主要為層級結構\n"
        else:
            report += f"  混合文檔 ({analysis['positive_ratio']:.1%}) - 包含大量正文\n"
        
        if bert_stats['count'] > 0:
            if bert_stats['mean'] > 0.8:
                report += f"  BERT高信心 ({bert_stats['mean']:.3f}) - 清晰的類別邊界\n"
            elif bert_stats['mean'] > 0.6:
                report += f"  BERT中等信心 ({bert_stats['mean']:.3f}) - 存在邊界情況\n"
            else:
                report += f"  BERT低信心 ({bert_stats['mean']:.3f}) - 需要檢查數據質量\n"
        
        if hierarchy and hierarchy.get('total_levels', 0) > 0:
            report += f"  層級複雜度: {hierarchy['total_levels']} 層 - 文檔結構化程度指標\n"
        
        report += f"  三層過濾策略 - 平衡了準確性、效率和可靠性\n"
        
        return report
    
    def save_results(self, output_file: str):
        """保存檢測結果 - 包含 numpy 類型轉換和層級分析"""
        results_data = []
        for result in self.detection_results:
            results_data.append({
                'line_number': result.line_number,
                'line_text': result.line_text,
                'detected_symbol': result.detected_symbol,
                'symbol_category': result.symbol_category,
                'rule_based_score': result.rule_based_score,
                'bert_score': result.bert_score,
                'final_prediction': result.final_prediction,
                'method_used': result.method_used
            })
        
        # 獲取層級分析
        hierarchy_analysis = self.detect_hierarchy_levels()
        
        output_data = {
            'detection_method': 'three_layer_hybrid_with_hierarchy',
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path) if self.model_path else None,
            'bert_model_loaded': self.is_model_loaded(),
            'total_lines': len(self.detection_results),
            'positive_predictions': sum(1 for r in self.detection_results if r.final_prediction),
            'ultra_strict_count': sum(1 for r in self.detection_results if r.method_used == "ultra_strict_pua"),
            'hierarchy_analysis': hierarchy_analysis,
            'results': results_data
        }
        
        # 決方案：序列化前統一轉換 numpy 類型
        output_data = convert_numpy_types(output_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 檢測結果已保存: {output_file}")
        
        # 額外保存層級結構到獨立文件
        if hierarchy_analysis and hierarchy_analysis.get('hierarchy_levels'):
            hierarchy_file = output_file.replace('.json', '_hierarchy.json')
            hierarchy_data = convert_numpy_types(hierarchy_analysis)
            
            with open(hierarchy_file, 'w', encoding='utf-8') as f:
                json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)
            
            print(f"🏗️ 層級結構已保存: {hierarchy_file}")

def main():
    """主函數 - 演示三層混合檢測器"""
    print("🚀 啟動三層混合層級符號檢測器 (推理專用)")
    print("'分層過濾' 原則：嚴格 → 軟規則 → 聚合")
    print("="*60)
    
    # 初始化檢測器
    detector = HybridLevelSymbolDetector()
    
    # 檢查是否有已訓練的模型
    model_path = "models/bert/level_detector/best_model"
    if Path(model_path).exists():
        print("📦 載入已訓練的 BERT 模型...")
        detector.load_bert_model(model_path)
    else:
        print("⚠️ 未找到 BERT 模型，將只使用規則檢測")
        print(f"💡 要訓練 BERT 模型，請運行: python train_bert_classifier.py")
    
    # 測試檢測
    test_file = "data/sample/TPDM,111,易,564,20250113,1.json"
    if Path(test_file).exists():
        print(f"\n🧪 測試檢測: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_lines = data['JFULL'].split('\n')
        
        # 執行三層混合檢測
        results = detector.detect_hybrid_markers(text_lines)
        
        # 分析結果
        analysis = detector.analyze_detection_results()
        
        # 生成報告
        report = detector.generate_detection_report(analysis)
        print(report)
        
        # 保存結果
        detector.save_results("three_layer_detection_results.json")
        
        # 顯示前10個正類結果
        positive_results = [r for r in results if r.final_prediction][:10]
        if positive_results:
            print(f"\n📋 檢測到的層級符號 (前10個):")
            for i, result in enumerate(positive_results, 1):
                method_icon = "🎯" if result.method_used == "ultra_strict_pua" else "🤖"
                confidence_info = f" (BERT: {result.bert_score:.3f})" if result.bert_score > 0 and result.bert_score < 1.0 else ""
                print(f"  {i:2}. {method_icon} 行 {result.line_number:3}: {result.detected_symbol}{confidence_info} - {result.line_text[:80]}...")
    
    else:
        print(f"❌ 測試文件不存在: {test_file}")

if __name__ == "__main__":
    main()
