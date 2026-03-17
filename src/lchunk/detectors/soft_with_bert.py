#!/usr/bin/env python3
"""
三層混合層級符號檢測器 (推理專用)
"分層過濾" 原則：嚴格 → 軟規則 → 聚合

三層策略：
1. 嚴格規則：PUA字符 + 頓號 = 100% 確定  (由 UltraStrictDetector 於管線上游完成)
2. 軟規則 + BERT：其他符號需要語義驗證
3. 最終聚合：合併所有檢測結果

注意：此版本只負責推理，不包含訓練功能
訓練請使用 train_bert_classifier.py
"""
import dotenv
dotenv.load_dotenv()  # 從 .env 文件加載環境變量

import json
import re
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

# 管線資料結構
from src.lchunk.pipeline import JudgmentArtifact, SymbolDetection

# BERT相關
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def convert_numpy_types(obj):
    """轉換 numpy 類型為 Python 原生類型"""
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
        
        # 如果提供了模型路徑，直接載入（支援本地路徑與 HuggingFace model ID）
        if model_path is not None:
            self.load_bert_model(model_path)
    
    def load_bert_model(self, model_path: Optional[str] = None):
        """載入訓練好的 BERT 模型（支援本地路徑與 HuggingFace model ID）"""
        if model_path:
            _path_str = str(model_path)
        elif self.model_path:
            _path_str = str(self.model_path)
        else:
            raise ValueError("請提供模型路徑")

        # 判斷是本地路徑還是 HuggingFace model ID
        _local = Path(_path_str)
        if _local.exists():
            model_identifier = _local
            print(f"📦 載入本地 BERT 模型: {model_identifier}")
        else:
            # 非本地路徑，視為 HuggingFace Hub model ID
            model_identifier = _path_str
            print(f"📦 從 HuggingFace Hub 載入 BERT 模型: {model_identifier}")

        self.model_path = _local  # 保持 self.model_path 相容性

        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_identifier)
        
        # 自適應設備選擇 - 優先使用 GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = self.bert_model.to(device)
        self.bert_model.eval()
        
        print(f"✅ BERT 模型載入完成 (device: {device})")
    
    def is_model_loaded(self) -> bool:
        """檢查 BERT 模型是否已載入"""
        return self.bert_model is not None and self.bert_tokenizer is not None
    
    # 大寫國字數字字元集（用於軟規則檢測）
    _UPPERCASE_CHN_DIGITS = frozenset('壹貳參肆伍陸柒捌玖拾佰千萬億')

    def _get_line_symbol_info(self, line_text: str) -> Tuple[bool, str, str]:
        """檢查行是否以層級符號開頭（軟規則）

        軟規則對映（規格 §4.3.2）：
          valid_symbol_ranges                              → 各類別（需後接頓號）
          r'^\\d'                                         → ARA_NUM
          r'^\\([一二三四五六七八九十]\\)'                → PAREN_CHN
          r'^\\(\\d+\\)'                                  → PAREN_ARA
          r'^\\([IVXLCDM]+\\)'                            → PAREN_ROMAN
          r'^[壹貳參肆伍陸柒捌玖拾佰千萬億]+'            → UPPERCASE_CHN_NUM

        注意：PUA 符號已在上游 UltraStrictDetector 處理，此處不再重複檢測。
        所有軟規則一律要求符號後方必須接頓號（、）。
        """
        if not line_text or len(line_text.strip()) == 0:
            return False, None, None

        stripped = line_text.strip()
        first_char = stripped[0]

        # ── 預定義符號範圍（valid_symbol_ranges）──────────────────────────────
        # 要求符號後方必須接頓號（、）
        for category, info in self.rule_detector.valid_symbol_ranges.items():
            if first_char in info["chars"]:
                if len(stripped) > 1 and stripped[1] == '、':
                    return True, first_char, category

        # ── 軟規則：半形阿拉伯數字（r'^\d'）─────────────────────────────────
        if re.match(r'^\d', stripped):
            if len(stripped) > 1 and stripped[1] == '、':
                return True, first_char, "ARA_NUM"

        # ── 軟規則：括號中文數字（r'^\([一二三四五六七八九十]\)'）─────────────
        paren_chn = re.match(r'^(\([一二三四五六七八九十]\))', stripped)
        if paren_chn:
            sym = paren_chn.group(1)
            if len(stripped) > len(sym) and stripped[len(sym)] == '、':
                return True, sym, "PAREN_CHN"

        # ── 軟規則：括號阿拉伯數字（r'^\(\d+\)'）────────────────────────────
        paren_ara = re.match(r'^(\(\d+\))', stripped)
        if paren_ara:
            sym = paren_ara.group(1)
            if len(stripped) > len(sym) and stripped[len(sym)] == '、':
                return True, sym, "PAREN_ARA"

        # ── 軟規則：括號羅馬數字（r'^\([IVXLCDM]+\)'）───────────────────────
        paren_roman = re.match(r'^(\([IVXLCDM]+\))', stripped)
        if paren_roman:
            sym = paren_roman.group(1)
            if len(stripped) > len(sym) and stripped[len(sym)] == '、':
                return True, sym, "PAREN_ROMAN"

        # ── 軟規則：大寫國字數字（壹貳參肆伍陸柒捌玖拾…）─────────────────────
        # 匹配連續的大寫國字數字，後接頓號（、）
        uppercase_chn = re.match(r'^([壹貳參肆伍陸柒捌玖拾佰千萬億]+)', stripped)
        if uppercase_chn:
            sym = uppercase_chn.group(1)
            if len(stripped) > len(sym) and stripped[len(sym)] == '、':
                return True, sym, "UPPERCASE_CHN_NUM"

        return False, None, None
    
    def bert_classify_lines(self, lines: List[str]) -> List[Tuple[float, int]]:
        """使用 BERT 對行進行分類 - 優化 GPU 推理性能"""
        if not self.is_model_loaded():
            raise ValueError("請先載入 BERT 模型")
        
        if not lines:
            return []
        
        # 自適應設備選擇
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = self.bert_model.to(device)
        
        # 準備輸入
        inputs = self.bert_tokenizer(
            lines,
            truncation=True,
            padding=True,
            max_length=128,
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
        
        # 返回結果到 CPU
        return list(zip(scores.cpu().numpy(), predictions.cpu().numpy()))

    # ==========================================
    # Pipeline-integrated method
    # ==========================================

    def process_artifact(self, artifact: JudgmentArtifact, bert_threshold: float = 0.5) -> None:
        """Pipeline 入口：處理 JudgmentArtifact，為尚未被 UltraStrict 標記的行進行軟規則 + BERT 分類。
        
        結果直接寫入每個 DocumentLine.detection。
        
        此方法假設 UltraStrictDetector.process_artifact() 已在上游執行完畢，
        因此已有 detection.method_used == "ultra_strict_pua" 的行將被跳過。
        """
        soft_candidates = []   # (doc_line, symbol, category)
        
        for doc_line in artifact.full_lines:
            # 跳過已被標記的行 (UltraStrict 結果或空行)
            if doc_line.detection is not None:
                continue
            
            stripped = doc_line.original_text.strip()
            if not stripped:
                doc_line.detection = SymbolDetection(method_used="empty_line")
                continue
            
            is_symbol_line, soft_symbol, soft_category = self._get_line_symbol_info(stripped)
            
            if is_symbol_line:
                soft_candidates.append((doc_line, soft_symbol, soft_category))
            else:
                doc_line.detection = SymbolDetection(method_used="rule_rejected")
        
        # BERT 批次分類
        if soft_candidates and self.is_model_loaded():
            candidate_texts = [dl.original_text.strip() for dl, _, _ in soft_candidates]
            bert_results = self.bert_classify_lines(candidate_texts)
            
            bert_accepted = 0
            for (doc_line, symbol, category), (bert_score, bert_prediction) in zip(soft_candidates, bert_results):
                final_prediction = float(bert_score) >= bert_threshold
                doc_line.detection = SymbolDetection(
                    detected_symbol=symbol if final_prediction else None,
                    symbol_category=category if final_prediction else None,
                    is_pua=self.rule_detector.is_pua_character(symbol) if symbol else False,
                    method_used="soft_rule_bert",
                    rule_based_score=0.5,
                    bert_score=float(bert_score),
                )
                if final_prediction:
                    bert_accepted += 1
            
            artifact.processing_stats['soft_rule_bert_accepted'] = bert_accepted
            artifact.processing_stats['soft_rule_bert_rejected'] = len(soft_candidates) - bert_accepted
        
        elif soft_candidates:
            # 沒有 BERT 模型，軟規則候選全部接受
            for doc_line, symbol, category in soft_candidates:
                doc_line.detection = SymbolDetection(
                    detected_symbol=symbol,
                    symbol_category=category,
                    is_pua=self.rule_detector.is_pua_character(symbol) if symbol else False,
                    method_used="soft_rule_only",
                    rule_based_score=0.5,
                    bert_score=0.0,
                )
            artifact.processing_stats['soft_rule_only_accepted'] = len(soft_candidates)
        
        artifact.processing_stats['soft_rule_candidates'] = len(soft_candidates)


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
        
        output_data = convert_numpy_types(output_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 檢測結果已保存: {output_file}")
        
        if hierarchy_analysis and hierarchy_analysis.get('hierarchy_levels'):
            hierarchy_file = output_file.replace('.json', '_hierarchy.json')
            hierarchy_data = convert_numpy_types(hierarchy_analysis)
            
            with open(hierarchy_file, 'w', encoding='utf-8') as f:
                json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)
            
            print(f"🏗️ 層級結構已保存: {hierarchy_file}")