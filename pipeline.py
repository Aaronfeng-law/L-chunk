"""
統一的 Pipeline 互動接口

提供以下功能：
1. 執行核心處理管線 (Splitter -> UltraStrict -> Hybrid -> Adaptive)
2. 匯出 Markdown，底層呼叫 src.lchunk.converters.md_exporter
"""
from src.lchunk.pipeline import (
    DocumentLine,
    SectionContent,
    JudgmentArtifact,
    PipelineOrchestrator
)
from src.lchunk.analyzers.splitter_refactor import split_judgment_document
# 作為可選模組匯入 md_exporter
try:
    from src.lchunk.converters.md_exporter import export_to_markdown
    _MD_AVAILABLE = True
except ImportError:
    _MD_AVAILABLE = False

import json
import argparse
from pathlib import Path


def process_single_file(input_path: Path, args: argparse.Namespace) -> None:
    """處理單一檔案並印出檢測結果與除錯資訊。"""
    if args.full:
        print(f"🚀 執行全管線檢測: {input_path.name}")
        orchestrator = PipelineOrchestrator(model_path=args.model_path)
        artifact = orchestrator.process_file(input_path)
    else:
        print(f"🔍 執行 Phase 1 (Splitter Only): {input_path.name}")
        artifact = split_judgment_document(input_path)

    if not artifact:
        print(f"❌ 無法解析檔案: {input_path}")
        return

    print("-" * 40)
    print(f"檔案: {artifact.file_path}")
    print(f"總行數: {len(artifact.full_lines)}")
    print(f"找到段落: {list(artifact.sections.keys())}")
    for section_name, content in artifact.sections.items():
        if content.lines:
            print(f"  [{section_name:15}] 包含 {len(content.lines):3} 行")

    if artifact.metadata:
        print(f"法院代碼: {artifact.metadata.get('JID', 'N/A')[:4]}")
        print(f"法院名稱: {artifact.metadata.get('court_full_name', 'Unknown')}")

    if args.full:
        detected_count = sum(1 for line in artifact.full_lines if line.detection and line.detection.detected_symbol)
        print(f"檢測到符號數: {detected_count}")
        if artifact.hierarchy_tree:
            print(f"層級樹節點數: {len(artifact.hierarchy_tree)}")

    if args.save:
        output_path = Path("output/debug")
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{input_path.stem}_debug.json"
        
        # 收集檢測到的符號標記
        markers = []
        for line in artifact.full_lines:
            if line.detection and line.detection.method_used not in ("", "empty_line", "rule_rejected"):
                markers.append({
                    "line": line.index,
                    "text": line.original_text.strip(),
                    "symbol": line.detection.detected_symbol,
                    "category": line.detection.symbol_category,
                    "level": line.detection.assigned_level,
                    "method": line.detection.method_used,
                    "bert_score": round(line.detection.bert_score, 4) if line.detection.bert_score else 0
                })

        result = {
            "metadata": artifact.metadata,
            "processing_stats": artifact.processing_stats,
            "hierarchy_tree": artifact.hierarchy_tree,
            "learned_rules": artifact.learned_rules,
            "sections": {k: len(v.lines) for k, v in artifact.sections.items()},
            "markers": markers
        }
        
        with open(output_path / file_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 結果已儲存至: {output_path / file_name} (包含 {len(markers)} 個檢測標記)")


def main():
    parser = argparse.ArgumentParser(
        description="法律文檔層級符號檢測系統 - 統一處理接口",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="輸入路徑：單一 JSON 檔案或包含多個 JSON 的目錄")
    parser.add_argument("--save", action="store_true", help="是否保存處理結果(debug JSON)")
    parser.add_argument("--full", action="store_true", help="執行全管線 (Splitter -> UltraStrict -> Hybrid -> Adaptive)")
    parser.add_argument("--debug", action="store_true", help="啟用除錯模式")
    parser.add_argument("--model-path", type=str, default="models/bert/level_detector/best_model", help="BERT 模型路徑")
    
    # Markdown 模式參數
    md_group = parser.add_argument_group("Markdown 匯出選項")
    md_group.add_argument("--markdown", action="store_true", help="將輸入轉換為 Markdown 格式（覆蓋一般資訊輸出行為）")
    md_group.add_argument("--md-output-dir", type=str, default="output/markdown", help="Markdown 輸出目錄（預設: output/markdown）")
    md_group.add_argument("--md-from-debug", action="store_true", help="讀取保存的 debug JSON 生成 Markdown，不再經過模型")
    md_group.add_argument("--md-legacy", action="store_true", help="使用舊版 AdaptiveHybridDetector 轉換（相容舊版結構）")
    md_group.add_argument("--md-machine", action="store_true", help="輸入視為舊版 machine export JSON 格式")
    md_group.add_argument("--max-files", type=int, default=None, help="目錄模式下最大處理檔案數量")

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"❌ Error: 路徑 {args.input_path} 不存在。")
        return

    # 若啟用了 Markdown 自動匯出
    if args.markdown:
        if not _MD_AVAILABLE:
            print("❌ Markdown 功能無法使用，可能缺少依賴或無法載入 src.lchunk.converters.md_exporter 模組。")
            return
            
        print("開始 Markdown 匯出程序...")
        export_to_markdown(
            input_path=input_path,
            output_dir=Path(args.md_output_dir),
            model_path=Path(args.model_path),
            max_files=args.max_files,
            use_pipeline=not args.md_legacy,
            from_debug_json=args.md_from_debug,
            machine_input=args.md_machine
        )
        return

    # 否則執行普通的 pipeline 檢測（支援單一檔案或目錄）
    if input_path.is_file():
        process_single_file(input_path, args)
    else:
        # 目錄模式
        candidates = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if args.max_files is not None:
            candidates = candidates[:args.max_files]
        
        if not candidates:
            print("找不到任何 JSON 檔案。")
            return
            
        for c in candidates:
            process_single_file(c, args)
            print("=" * 60)

if __name__ == "__main__":
    main()