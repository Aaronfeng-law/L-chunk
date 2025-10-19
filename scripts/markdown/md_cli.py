#!/usr/bin/env python3
"""CLI interface for converting adaptive detection results into Markdown files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lchunk.converters.md_converter import MarkdownConverter  # pylint: disable=wrong-import-position
from src.lchunk.detectors.adaptive_hybrid import AdaptiveHybridDetector  # pylint: disable=wrong-import-position


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate adaptive detector results into Markdown."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single JSON judgment file or a directory containing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "markdown",
        help="Directory where generated Markdown files will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "level_detector" / "best_model",
        help="Optional path to the fine tuned level detector model.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of files to process when input_path is a directory.",
    )
    parser.add_argument(
        "--machine-input",
        action="store_true",
        help="Treat input JSON as detector machine exports (skip rerunning the detector).",
    )
    return parser.parse_args(list(argv))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_targets(input_path: Path, max_files: int | None) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates = sorted(p for p in input_path.glob("*.json") if p.is_file())
    if max_files is not None:
        return candidates[:max_files]
    return candidates


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    ensure_output_dir(args.output_dir)

    detector: AdaptiveHybridDetector | None = None
    if not args.machine_input:
        detector = AdaptiveHybridDetector(str(args.model_path) if args.model_path.exists() else None)
    
    converter = MarkdownConverter(detector)

    targets = collect_targets(args.input_path, args.max_files)
    if not targets:
        print("No JSON files found to process.")
        return 0

    # Process files individually to show progress
    written_files: List[Path] = []
    for path in targets:
        print(f"Converting {path} -> Markdown")
        try:
            output_path = converter.convert_to_markdown(path, args.output_dir, args.machine_input)
            if output_path is not None:
                written_files.append(output_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to convert {path.name}: {exc}")
            continue

    print(f"Generated {len(written_files)} Markdown file(s) in {args.output_dir}")
    for item in written_files:
        print(f" - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
