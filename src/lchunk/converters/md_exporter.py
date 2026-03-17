"""Markdown exporter module.

提供將判決文件轉換為 Markdown 格式的功能。
支援兩種模式：
  1. 新版 Pipeline 模式（Hierarchy Tree）
  2. 舊版相容模式（AdaptiveHybridDetector）
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.lchunk.converters.md_converter import (
    PipelineMarkdownConverter,
    MarkdownConverter,
    _LEGACY_AVAILABLE,
)

def collect_targets(input_path: Path, max_files: int | None = None) -> List[Path]:
    """收集待處理的 JSON 檔案清單。"""
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"輸入路徑不存在: {input_path}")
    candidates = sorted(p for p in input_path.glob("*.json") if p.is_file())
    if max_files is not None:
        return candidates[:max_files]
    return candidates

def export_to_markdown(
    input_path: Path,
    output_dir: Path,
    *,
    model_path: Optional[Path] = None,
    max_files: Optional[int] = None,
    use_pipeline: bool = True,
    from_debug_json: bool = False,
    machine_input: bool = False,
) -> int:
    """轉換 JSON 檔案為 Markdown 並儲存至輸出目錄。

    Args:
        input_path: 來源路徑（檔案或目錄）
        output_dir: 輸出目標目錄
        model_path: BERT 模型路徑（若是 pipeline 以外的原始處理）
        max_files: 目錄模式下最多處理幾個檔案
        use_pipeline: 是否使用新版 PipelineMarkdownConverter
        from_debug_json: 是否從 debug JSON 直接生成 (pipeline 模式)
        machine_input: 是否為舊版 machine export JSON
    
    Returns:
        成功產生的 Markdown 檔案數量
    """
    targets = collect_targets(input_path, max_files)
    if not targets:
        print("找不到任何 JSON 檔案。")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_pipeline:
        print("🚀 使用新版 PipelineMarkdownConverter")
        if from_debug_json:
            print("   模式: 讀取 debug JSON（hierarchy_tree）")
        else:
            print("   模式: 跑完整 Pipeline（Splitter→UltraStrict→Hybrid→Adaptive）")
            if model_path and model_path.exists():
                print(f"   BERT 模型: {model_path}")
            else:
                print(f"   ⚠️  BERT 模型不存在，將使用純規則模式: {model_path}")

        converter = PipelineMarkdownConverter()
        written = converter.convert_batch(
            targets,
            output_dir,
            model_path=model_path if not from_debug_json else None,
            from_debug_json=from_debug_json,
        )
    else:
        if not _LEGACY_AVAILABLE:
            print("❌ 舊版模式需要 AdaptiveHybridDetector，但匯入失敗。")
            print("   請改用 --pipeline 模式。")
            return 0

        print("🔧 使用舊版 MarkdownConverter（AdaptiveHybridDetector）")
        detector = None
        if not machine_input:
            from lchunk.detectors.adaptive_hierarchy import AdaptiveHybridDetector  # type: ignore
            detector = AdaptiveHybridDetector(
                str(model_path) if model_path and model_path.exists() else None
            )

        converter_legacy = MarkdownConverter(detector)
        written = []
        for path in targets:
            print(f"Converting {path.name} → Markdown")
            try:
                out = converter_legacy.convert_to_markdown(path, output_dir, machine_input)
                if out is not None:
                    written.append(out)
            except Exception as exc:
                print(f"  ❌ 失敗 {path.name}: {exc}")

    print(f"\n✅ 共產出 {len(written)} 個 Markdown 檔案 → {output_dir}")
    for item in written:
        print(f"   - {item.name}")

    return len(written)
