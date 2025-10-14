#!/usr/bin/env python3
"""Convert line based chunking results into Markdown files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lchunk.detectors.adaptive_hybrid import (  # pylint: disable=wrong-import-position
    IntelligentDetectionResult,
    IntelligentHybridDetector,
    LineBasedChunk,
)


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


def sanitize_lines(lines: Iterable[str]) -> List[str]:
    return [line.strip() for line in lines if line and line.strip()]


def format_level_block(
    chunk: LineBasedChunk,
    numbering_state: Dict[int, int],
    content_lines: List[str],
) -> List[str]:
    if chunk.level <= 0:
        raise ValueError("format_level_block expects level >= 1")

    numbering_state.setdefault(chunk.level, 0)
    numbering_state[chunk.level] += 1

    for deeper_level in [lvl for lvl in numbering_state if lvl > chunk.level]:
        numbering_state.pop(deeper_level, None)

    indent = "    " * max(chunk.level - 1, 0)
    first_line = f"{indent}{numbering_state[chunk.level]}. {content_lines[0]}"
    formatted = [first_line]

    continuation_indent = f"{indent}   " if indent else "   "
    for extra_line in content_lines[1:]:
        formatted.append(f"{continuation_indent}{extra_line}")

    return formatted


def chunks_to_markdown(result: IntelligentDetectionResult) -> str:
    if not result.line_based_chunks:
        raise ValueError("Detection result does not contain line based chunks.")

    output_lines: List[str] = [f"# {result.filename}", f"- learning_region: {result.learning_region}", ""]
    numbering_state: Dict[int, int] = {}

    for chunk in sorted(result.line_based_chunks, key=lambda item: item.start_line):
        content_lines = sanitize_lines(chunk.content_lines)
        if not content_lines:
            continue

        if chunk.level == 0:
            output_lines.append(f"## {' '.join(content_lines)}")
            output_lines.append("")
            numbering_state.clear()
        elif chunk.level >= 1:
            output_lines.extend(format_level_block(chunk, numbering_state, content_lines))
            output_lines.append("")
        elif chunk.level == -1:
            output_lines.extend(content_lines)
            output_lines.append("")
        elif chunk.level == -2:
            output_lines.append(f"*{content_lines[0]}*")
            output_lines.append("")
        elif chunk.level == -3:
            output_lines.extend(content_lines)
            output_lines.append("")

    cleaned_output = [line.rstrip() for line in output_lines]
    return "\n".join(cleaned_output).strip() + "\n"


def process_file(detector: IntelligentHybridDetector, input_file: Path, output_dir: Path) -> Path | None:
    result = detector.process_single_file(input_file)
    if not result:
        return None

    markdown = chunks_to_markdown(result)
    target_path = output_dir / f"{input_file.stem}.md"
    target_path.write_text(markdown, encoding="utf-8")
    return target_path


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    ensure_output_dir(args.output_dir)

    detector = IntelligentHybridDetector(str(args.model_path) if args.model_path.exists() else None)

    targets = collect_targets(args.input_path, args.max_files)
    if not targets:
        print("No JSON files found to process.")
        return 0

    written_files: List[Path] = []
    for path in targets:
        print(f"Converting {path} -> Markdown")
        try:
            output_path = process_file(detector, path, args.output_dir)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to convert {path.name}: {exc}")
            continue

        if output_path is not None:
            written_files.append(output_path)

    print(f"Generated {len(written_files)} Markdown file(s) in {args.output_dir}")
    for item in written_files:
        print(f" - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
