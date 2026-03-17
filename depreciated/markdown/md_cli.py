#!/usr/bin/env python3
"""CLI interface for converting judgment documents into Markdown files.

支援兩種模式：
  --pipeline   使用新版 PipelineMarkdownConverter（JudgmentArtifact / hierarchy_tree）
               - 預設：對輸入 JSON 跑完整 Pipeline（Splitter→UltraStrict→Hybrid→Adaptive）
               - 搭配 --from-debug-json：讀取 pipeline.py --save 所產生的 debug JSON
  （預設）     使用舊版 MarkdownConverter（AdaptiveHybridDetector，向後相容）
               - 搭配 --machine-input：讀取舊版 machine export JSON

用法範例：
    # 新版 Pipeline 模式（跑完整 Pipeline）：
    python scripts/markdown/md_cli.py data/ --pipeline

    # 新版 Pipeline 模式（讀已有的 debug JSON）：
    python scripts/markdown/md_cli.py output/debug/ --pipeline --from-debug-json

    # 舊版相容模式：
    python scripts/markdown/md_cli.py data/ --machine-input
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lchunk.converters.md_converter import (  # pylint: disable=wrong-import-position
    PipelineMarkdownConverter,
    MarkdownConverter,
    _LEGACY_AVAILABLE,
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將判決文件轉換為 Markdown 格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="輸入來源：單一 JSON 檔案或包含多個 JSON 的目錄。",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=PROJECT_ROOT / "output" / "markdown",
        help="Markdown 輸出目錄（預設: output/markdown）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "level_detector" / "best_model",
        help="BERT 模型路徑（僅 pipeline 模式使用）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="目錄模式下最多處理幾個檔案",
    )

    # ── 模式選擇 ──────────────────────────────────────────────────
    mode_group = parser.add_argument_group("模式選擇（二擇一）")
    mode_group.add_argument(
        "--pipeline",
        action="store_true",
        help="【推薦】使用新版 PipelineMarkdownConverter（hierarchy_tree 巢狀樹）",
    )
    mode_group.add_argument(
        "--machine-input",
        action="store_true",
        help="【舊版相容】將輸入視為 AdaptiveHybridDetector machine export JSON",
    )

    # ── Pipeline 子選項 ───────────────────────────────────────────
    pipeline_group = parser.add_argument_group("Pipeline 模式選項（搭配 --pipeline 使用）")
    pipeline_group.add_argument(
        "--from-debug-json",
        action="store_true",
        help="讀取 pipeline.py --save 輸出的 debug JSON，而非重跑 Pipeline",
    )

    return parser.parse_args(list(argv))


def collect_targets(input_path: Path, max_files: int | None) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"輸入路徑不存在: {input_path}")
    candidates = sorted(p for p in input_path.glob("*.json") if p.is_file())
    if max_files is not None:
        return candidates[:max_files]
    return candidates


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    targets = collect_targets(args.input_path, args.max_files)
    if not targets:
        print("找不到任何 JSON 檔案。")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # 新版 Pipeline 模式
    # ══════════════════════════════════════════════════════
    if args.pipeline:
        print("🚀 使用新版 PipelineMarkdownConverter")
        if args.from_debug_json:
            print("   模式: 讀取 debug JSON（hierarchy_tree）")
        else:
            print("   模式: 跑完整 Pipeline（Splitter→UltraStrict→Hybrid→Adaptive）")
            model_path = args.model_path
            if model_path.exists():
                print(f"   BERT 模型: {model_path}")
            else:
                print(f"   ⚠️  BERT 模型不存在，將使用純規則模式: {model_path}")

        converter = PipelineMarkdownConverter()
        written = converter.convert_batch(
            targets,
            args.output_dir,
            model_path=args.model_path if not args.from_debug_json else None,
            from_debug_json=args.from_debug_json,
        )

    # ══════════════════════════════════════════════════════
    # 舊版相容模式
    # ══════════════════════════════════════════════════════
    else:
        if not _LEGACY_AVAILABLE:
            print("❌ 舊版模式需要 AdaptiveHybridDetector，但匯入失敗。")
            print("   請改用 --pipeline 模式。")
            return 1

        print("🔧 使用舊版 MarkdownConverter（AdaptiveHybridDetector）")
        detector = None
        if not args.machine_input:
            try:
                from lchunk.detectors.adaptive_hierarchy import AdaptiveHybridDetector  # type: ignore
                detector = AdaptiveHybridDetector(
                    str(args.model_path) if args.model_path.exists() else None
                )
            except ImportError as e:
                print(f"❌ 無法載入 AdaptiveHybridDetector: {e}")
                return 1

        converter_legacy = MarkdownConverter(detector)
        written = []
        for path in targets:
            print(f"Converting {path.name} → Markdown")
            try:
                out = converter_legacy.convert_to_markdown(path, args.output_dir, args.machine_input)
                if out is not None:
                    written.append(out)
            except Exception as exc:
                print(f"  ❌ 失敗 {path.name}: {exc}")

    # ── 結果摘要 ─────────────────────────────────────────────────
    print(f"\n✅ 共產出 {len(written)} 個 Markdown 檔案 → {args.output_dir}")
    for item in written:
        print(f"   - {item.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
