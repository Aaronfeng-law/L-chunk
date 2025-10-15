"""Utilities for converting detector output into Markdown documents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..detectors.adaptive_hybrid import (
	IntelligentDetectionResult,
	LineBasedChunk,
)
from ..interfaces import DetectorService


def _sanitize_lines(lines: Iterable[str]) -> List[str]:
	"""Clean raw lines by stripping whitespace and dropping empties."""

	return [line.strip() for line in lines if line and line.strip()]


def _format_level_block(
	chunk: LineBasedChunk,
	numbering_state: Dict[int, int],
	content_lines: List[str],
) -> Tuple[List[str], str]:
	"""Format a level >= 1 chunk as a numbered Markdown list block."""

	if chunk.level <= 0:
		raise ValueError("format_level_block expects level >= 1")

	numbering_state.setdefault(chunk.level, 0)
	numbering_state[chunk.level] += 1

	# Reset deeper levels when ascending back up the hierarchy.
	for deeper_level in [lvl for lvl in list(numbering_state.keys()) if lvl > chunk.level]:
		numbering_state.pop(deeper_level, None)

	indent = "    " * max(chunk.level - 1, 0)
	first_line = f"{indent}{numbering_state[chunk.level]}. {content_lines[0]}"
	formatted = [first_line]

	content_indent = f"{indent}   " if indent else "   "
	for extra_line in content_lines[1:]:
		formatted.append(f"{content_indent}{extra_line}")

	return formatted, content_indent


def build_markdown_document(result: IntelligentDetectionResult) -> str:
	"""Convert a single detection result into Markdown text."""

	if not result.line_based_chunks:
		raise ValueError("Detection result does not contain line based chunks.")

	output_lines: List[str] = [
		f"# {result.filename}",
		f"- learning_region: {result.learning_region}",
		"",
	]
	numbering_state: Dict[int, int] = {}
	last_list_indent: str | None = None

	sorted_chunks = sorted(result.line_based_chunks, key=lambda item: item.start_line)

	for idx, chunk in enumerate(sorted_chunks):
		raw_lines = [line.rstrip("\n") for line in chunk.content_lines]

		if chunk.level == -3:
			header_footer_lines = [line for line in raw_lines if line.strip()]
			if not header_footer_lines:
				continue
			output_lines.extend(header_footer_lines)
			output_lines.append("")
			last_list_indent = None
			continue

		prev_chunk = sorted_chunks[idx - 1] if idx > 0 else None
		next_chunk = sorted_chunks[idx + 1] if idx + 1 < len(sorted_chunks) else None

		content_lines = _sanitize_lines(raw_lines)
		if not content_lines:
			continue

		if chunk.level == 0:
			heading_text = " ".join(content_lines)
			if not heading_text:
				continue
			output_lines.append(f"## {heading_text}")
			output_lines.append("")
			numbering_state.clear()
			last_list_indent = None
		elif chunk.level >= 1:
			list_payload = [" ".join(content_lines)] if content_lines else []
			if not list_payload:
				continue
			formatted_lines, content_indent = _format_level_block(
				chunk, numbering_state, list_payload
			)
			output_lines.extend(formatted_lines)
			last_list_indent = content_indent
			output_lines.append("")
		elif chunk.level == -1:
			adjacent_date = (
				(prev_chunk and prev_chunk.level == 0 and prev_chunk.chunk_type == "date")
				or (next_chunk and next_chunk.level == 0 and next_chunk.chunk_type == "date")
			)

			if adjacent_date:
				line_by_line = [line for line in raw_lines if line.strip()]
				if not line_by_line:
					continue
				output_lines.extend(line_by_line)
				output_lines.append("")
				last_list_indent = None
			else:
				paragraph = " ".join(content_lines)
				if not paragraph:
					continue
				if last_list_indent:
					output_lines.append(f"{last_list_indent}{paragraph}")
				else:
					output_lines.append(paragraph)
				output_lines.append("")
		elif chunk.level == -2:
			date_text = " ".join(content_lines)
			if not date_text:
				continue
			output_lines.append(f"*{date_text}*")
			output_lines.append("")
			last_list_indent = None

	cleaned_output = [line.rstrip() for line in output_lines]
	return "\n".join(cleaned_output).strip() + "\n"


def convert_detection_result(
	result: IntelligentDetectionResult,
	output_dir: Path,
) -> Path:
	"""Persist a detection result as Markdown inside *output_dir*."""

	markdown = build_markdown_document(result)
	output_dir.mkdir(parents=True, exist_ok=True)
	target_path = output_dir / f"{Path(result.filename).stem}.md"
	target_path.write_text(markdown, encoding="utf-8")
	return target_path


def collect_targets(input_path: Path, max_files: int | None = None) -> List[Path]:
	if input_path.is_file():
		return [input_path]

	if not input_path.exists():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")

	candidates = sorted(p for p in input_path.glob("*.json") if p.is_file())
	if max_files is not None:
		return candidates[:max_files]
	return candidates


def convert_inputs_to_markdown(
	detector_service: DetectorService,
	input_path: Path,
	output_dir: Path,
	max_files: int | None = None,
) -> List[Path]:
	"""Convert a JSON file or directory of files into Markdown outputs."""

	targets = collect_targets(input_path, max_files)
	if not targets:
		return []

	output_dir.mkdir(parents=True, exist_ok=True)

	written_files: List[Path] = []
	for target in targets:
		result = detector_service.detect_single(target)
		if not result:
			continue
		markdown_path = convert_detection_result(result, output_dir)
		written_files.append(markdown_path)
	return written_files


__all__ = [
	"build_markdown_document",
	"convert_detection_result",
	"convert_inputs_to_markdown",
	"collect_targets",
]
