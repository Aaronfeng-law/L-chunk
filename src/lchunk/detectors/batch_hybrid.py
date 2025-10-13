#!/usr/bin/env python3
"""
混合批次層級檢測器
基於 Linus "漸進式過濾" 原則的批次處理版本

結合：
1. 規則檢測：零容忍格式檢查
2. BERT分類：精確語義理解
3. 批次處理：高效處理大量文檔

Linus式設計原則：
- "Good programmers worry about data structures" - 優化批次數據流
- "Talk is cheap. Show me the code" - 實際測試證明效果
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入混合檢測器
from .hybrid import HybridLevelSymbolDetector, HybridDetectionResult

@dataclass
class HybridProcessingStats:
    """混合處理統計資料"""
    file_path: str
    file_name: str
    start_time: float
    end_time: float
    processing_time: float
    total_lines: int
    candidate_lines: int  # 規則檢測候選行數
    bert_processed_lines: int  # BERT處理行數
    total_markers: int
    ultra_strict_markers: int
    bert_refined_markers: int
    rule_only_markers: int
    success: bool
    error_message: str = ""
    bert_model_used: bool = False
    output_data: Optional[Dict] = None

class HybridBatchProcessor:
    """混合批次處理器 - Linus式高效設計"""
    
    def __init__(self, output_base_dir: str = "hybrid_output", model_path: str = None):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.processing_stats: List[HybridProcessingStats] = []
        self.batch_start_time = 0
        self.batch_end_time = 0
        
        # 初始化混合檢測器
        print("🔧 初始化混合檢測器...")
        self.detector = HybridLevelSymbolDetector(model_path)
        
        # 檢查 BERT 模型狀態
        if self.detector.is_model_loaded():
            print("✅ BERT 模型已載入，將使用混合檢測")
        else:
            print("⚠️ 未載入 BERT 模型，將只使用規則檢測")
        
    def process_single_file(self, file_path: Path) -> HybridProcessingStats:
        """處理單一檔案（混合檢測）"""
        start_time = time.time()
        stats = HybridProcessingStats(
            file_path=str(file_path),
            file_name=file_path.name,
            start_time=start_time,
            end_time=0,
            processing_time=0,
            total_lines=0,
            candidate_lines=0,
            bert_processed_lines=0,
            total_markers=0,
            ultra_strict_markers=0,
            bert_refined_markers=0,
            rule_only_markers=0,
            success=False,
            bert_model_used=self.detector.is_model_loaded()
        )
        
        try:
            # 載入 JSON 檔案
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'JFULL' not in data:
                stats.error_message = "檔案中沒有 'JFULL' 欄位"
                return stats
            
            # 分割文本行
            text_lines = data['JFULL'].split('\n')
            stats.total_lines = len(text_lines)
            
            # 執行三層混合檢測
            results = self.detector.detect_hybrid_markers(text_lines)
            
            # 統計結果
            stats.total_markers = sum(1 for r in results if r.final_prediction)
            stats.ultra_strict_markers = sum(1 for r in results if r.method_used == "ultra_strict_pua")
            stats.bert_refined_markers = sum(1 for r in results if r.final_prediction and r.method_used == "soft_rule_bert")
            stats.rule_only_markers = sum(1 for r in results if r.final_prediction and r.method_used in ["ultra_strict_pua", "soft_rule_only"])
            stats.candidate_lines = sum(1 for r in results if r.rule_based_score > 0)
            stats.bert_processed_lines = sum(1 for r in results if r.bert_score > 0 and r.bert_score < 1.0)  # 排除終極嚴格的1.0分
            
            # 分析結果
            analysis = self.detector.analyze_detection_results()
            
            # 準備輸出數據
            markers_data = []
            for result in results:
                if result.final_prediction:
                    markers_data.append({
                        "line_number": result.line_number,
                        "symbol": result.detected_symbol,
                        "unicode_code": f"U+{ord(result.detected_symbol):04X}" if result.detected_symbol else None,
                        "category": result.symbol_category,
                        "content": result.line_text,
                        "rule_score": result.rule_based_score,
                        "bert_score": result.bert_score,
                        "method": result.method_used,
                        "is_pua": self.detector.rule_detector.is_pua_character(result.detected_symbol) if result.detected_symbol else False
                    })
            
            output_data = {
                "source_file": str(file_path),
                "detection_method": "hybrid_rule_bert",
                "bert_model_used": stats.bert_model_used,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0,  # 將在後面更新
                "statistics": {
                    "total_lines": stats.total_lines,
                    "candidate_lines": stats.candidate_lines,
                    "bert_processed_lines": stats.bert_processed_lines,
                    "total_markers": stats.total_markers,
                    "ultra_strict_markers": stats.ultra_strict_markers,
                    "bert_refined_markers": stats.bert_refined_markers,
                    "rule_only_markers": stats.rule_only_markers
                },
                "analysis": analysis,
                "markers": markers_data
            }
            
            stats.output_data = output_data
            stats.success = True
            
        except FileNotFoundError:
            stats.error_message = "檔案不存在"
        except json.JSONDecodeError:
            stats.error_message = "JSON 檔案格式錯誤"
        except Exception as e:
            stats.error_message = f"處理錯誤: {str(e)}"
        
        # 更新時間統計
        stats.end_time = time.time()
        stats.processing_time = stats.end_time - stats.start_time
        
        if stats.success and stats.output_data:
            stats.output_data["processing_time"] = stats.processing_time
        
        return stats
    
    def process_directory(self, input_dir: Path, output_subdir: str) -> Dict:
        """處理整個目錄"""
        print(f"🔍 掃描目錄: {input_dir}")
        
        # 查找所有JSON檔案
        json_files = list(input_dir.glob("*.json"))
        print(f"📁 找到 {len(json_files)} 個 JSON 檔案")
        
        if not json_files:
            return {"error": "沒有找到 JSON 檔案"}
        
        # 創建輸出子目錄
        output_dir = self.output_base_dir / output_subdir
        output_dir.mkdir(exist_ok=True)
        
        # 批次處理統計
        batch_stats = {
            "detection_method": "hybrid_rule_bert",
            "bert_model_used": self.detector.is_model_loaded(),
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_files": len(json_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0,
            "total_lines": 0,
            "total_candidate_lines": 0,
            "total_bert_processed_lines": 0,
            "total_markers": 0,
            "total_ultra_strict_markers": 0,
            "total_bert_refined_markers": 0,
            "total_rule_only_markers": 0,
            "files": []
        }
        
        self.batch_start_time = time.time()
        
        # 處理每個檔案
        for i, file_path in enumerate(json_files, 1):
            print(f"\n📄 處理檔案 {i}/{len(json_files)}: {file_path.name}")
            
            # 處理檔案
            stats = self.process_single_file(file_path)
            
            if stats.success:
                # 保存單個檔案結果
                output_file = output_dir / f"{file_path.stem}_hybrid_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(stats.output_data, f, ensure_ascii=False, indent=2)
                
                batch_stats["successful_files"] += 1
                batch_stats["total_lines"] += stats.total_lines
                batch_stats["total_candidate_lines"] += stats.candidate_lines
                batch_stats["total_bert_processed_lines"] += stats.bert_processed_lines
                batch_stats["total_markers"] += stats.total_markers
                batch_stats["total_ultra_strict_markers"] += stats.ultra_strict_markers
                batch_stats["total_bert_refined_markers"] += stats.bert_refined_markers
                batch_stats["total_rule_only_markers"] += stats.rule_only_markers
                
                print(f"✅ 成功 - 檢測到 {stats.total_markers} 個標記 (終極嚴格: {stats.ultra_strict_markers}, BERT: {stats.bert_refined_markers}, 軟規則: {stats.rule_only_markers})")
            else:
                batch_stats["failed_files"] += 1
                print(f"❌ 失敗 - {stats.error_message}")
            
            batch_stats["total_processing_time"] += stats.processing_time
            
            # 記錄檔案統計
            file_summary = {
                "file_name": stats.file_name,
                "success": stats.success,
                "processing_time": stats.processing_time,
                "total_lines": stats.total_lines,
                "candidate_lines": stats.candidate_lines,
                "bert_processed_lines": stats.bert_processed_lines,
                "total_markers": stats.total_markers,
                "bert_refined_markers": stats.bert_refined_markers,
                "rule_only_markers": stats.rule_only_markers,
                "error_message": stats.error_message
            }
            batch_stats["files"].append(file_summary)
            
            self.processing_stats.append(stats)
        
        self.batch_end_time = time.time()
        
        # 生成批次報告
        self._generate_hybrid_batch_report(batch_stats, output_dir)
        
        return batch_stats
    
    def _generate_hybrid_batch_report(self, batch_stats: Dict, output_dir: Path):
        """生成混合批次處理報告"""
        avg_time = batch_stats["total_processing_time"] / batch_stats["total_files"] if batch_stats["total_files"] > 0 else 0
        success_rate = (batch_stats["successful_files"] / batch_stats["total_files"] * 100) if batch_stats["total_files"] > 0 else 0
        
        # 計算效率指標
        candidate_rate = (batch_stats["total_candidate_lines"] / batch_stats["total_lines"] * 100) if batch_stats["total_lines"] > 0 else 0
        bert_refinement_rate = (batch_stats["total_bert_refined_markers"] / batch_stats["total_markers"] * 100) if batch_stats["total_markers"] > 0 else 0
        
        report = f"""
============================================================
⚡ 混合批次層級檢測報告 (Linus式漸進過濾)
============================================================

🔧 檢測配置:
  檢測方法: {batch_stats['detection_method']}
  BERT 模型: {'✅ 已載入' if batch_stats['bert_model_used'] else '❌ 未載入'}
  輸入目錄: {batch_stats['input_directory']}
  輸出目錄: {batch_stats['output_directory']}

📊 批次處理統計:
  總檔案數: {batch_stats['total_files']}
  成功處理: {batch_stats['successful_files']}
  處理失敗: {batch_stats['failed_files']}
  成功率: {success_rate:.1f}%

⏱️  時間統計:
  總處理時間: {batch_stats['total_processing_time']:.3f} 秒
  平均處理時間: {avg_time:.3f} 秒/檔案
  批次總時間: {self.batch_end_time - self.batch_start_time:.3f} 秒

🎯 檢測統計:
  總文本行數: {batch_stats['total_lines']:,}
  候選行數: {batch_stats['total_candidate_lines']:,} ({candidate_rate:.1f}%)
  BERT 處理行數: {batch_stats['total_bert_processed_lines']:,}
  
📈 檢測結果:
  總標記數: {batch_stats['total_markers']:,}
  BERT 精煉: {batch_stats['total_bert_refined_markers']:,} ({bert_refinement_rate:.1f}%)
  規則檢測: {batch_stats['total_rule_only_markers']:,}
  平均標記/檔: {batch_stats['total_markers'] / batch_stats['successful_files'] if batch_stats['successful_files'] > 0 else 0:.1f}

⚡ Linus式洞察:
  漸進過濾效率: {candidate_rate:.1f}% 的行進入 BERT 處理
  BERT 貢獻度: {bert_refinement_rate:.1f}% 的標記經過語義精煉
  整體性能: 平衡了準確性和計算效率

📋 檔案處理明細 (前20個):
"""
        
        # 按處理時間排序顯示檔案
        sorted_files = sorted(batch_stats["files"], key=lambda x: x["processing_time"], reverse=True)
        for i, file_info in enumerate(sorted_files[:20], 1):
            status = "✅" if file_info["success"] else "❌"
            bert_info = f" (BERT: {file_info['bert_refined_markers']})" if batch_stats['bert_model_used'] else ""
            report += f"  {i:2}. {status} {file_info['file_name']:40} | {file_info['processing_time']:.3f}s | {file_info['total_markers']} 標記{bert_info}\n"
        
        if len(sorted_files) > 20:
            report += f"      ... 還有 {len(sorted_files) - 20} 個檔案\n"
        
        print(report)
        
        # 保存詳細報告
        with open(output_dir / 'batch_hybrid_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存 JSON 統計
        with open(output_dir / 'batch_hybrid_stats.json', 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="混合批次層級檢測器")
    parser.add_argument("input_dir", help="輸入目錄路徑")
    parser.add_argument("--output", "-o", default="hybrid_output", help="輸出目錄 (默認: hybrid_output)")
    parser.add_argument("--subdir", "-s", help="輸出子目錄名稱 (默認: 使用輸入目錄名)")
    parser.add_argument("--model", "-m", help="BERT 模型路徑 (默認: bert_level_detector/best_model)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ 錯誤：輸入目錄不存在: {input_path}")
        return
    
    if not input_path.is_dir():
        print(f"❌ 錯誤：路徑不是目錄: {input_path}")
        return
    
    # 確定 BERT 模型路徑
    model_path = args.model
    if not model_path:
        default_model = Path("bert_level_detector/best_model")
        if default_model.exists():
            model_path = str(default_model)
        else:
            print("⚠️ 未找到預設 BERT 模型，將只使用規則檢測")
    
    # 確定輸出子目錄名稱
    output_subdir = args.subdir or input_path.name
    
    print("⚡ 啟動混合批次層級檢測器")
    print("基於 Linus '漸進式過濾' 原則")
    print("規則檢測 + BERT 語義理解 = 最佳準確性")
    print()
    
    # 初始化批次處理器
    processor = HybridBatchProcessor(args.output, model_path)
    
    # 執行批次處理
    batch_stats = processor.process_directory(input_path, output_subdir)
    
    if "error" in batch_stats:
        print(f"❌ 批次處理失敗: {batch_stats['error']}")
        return
    
    print(f"\n🎉 混合批次處理完成!")
    print(f"📊 處理了 {batch_stats['successful_files']}/{batch_stats['total_files']} 個檔案")
    print(f"⚡ 發現 {batch_stats['total_markers']} 個標記")
    if batch_stats['bert_model_used']:
        print(f"🤖 BERT 精煉了 {batch_stats['total_bert_refined_markers']} 個標記")
    print(f"💾 結果保存在: {processor.output_base_dir / output_subdir}")

if __name__ == "__main__":
    main()