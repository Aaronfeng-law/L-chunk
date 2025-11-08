#!/usr/bin/env python3
"""Command line entry point for the adaptive hybrid detector."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lchunk.detectors.adaptive_hybrid import (  # pylint: disable=wrong-import-position
    AdaptiveDetectionResult,
    AdaptiveHybridDetector,
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the adaptive hybrid detector on a JSON file or directory of files.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a JSON judgment file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "bert" / "level_detector" / "best_model",
        help="Optional fine-tuned level detector to load before inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "adaptive",
        help="Directory where reports and serialized outputs will be written.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "adaptive_detector.log",
        help="File to write detailed execution logs.",
    )
    parser.add_argument(
        "--export-format",
        choices=("human", "machine"),
        default="machine",
        help=(
            "Select 'human' for Markdown/summary reports or 'machine' for JSON exports "
            "with level -1 content merged into upper levels (excluding level 0)."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="When processing a directory, limit the number of files scanned (sorted alphabetically).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate human-readable Markdown reports in addition to machine-readable JSON exports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also emit progress information to the console (default keeps console quiet).",
    )
    return parser.parse_args(list(argv))


def configure_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    for noisy_logger in ("transformers", "urllib3", "matplotlib"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def summarize_result(result: AdaptiveDetectionResult) -> str:
    symbol_count = len([item for item in result.full_detection_results if item.final_prediction])
    return (
        f"learning_region={result.learning_region}, "
        f"learned_rules={len(result.learned_rules)}, "
        f"symbols={symbol_count}, "
        f"line_chunks={len(result.line_based_chunks or [])}"
    )


def run(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, args.verbose)

    logger = logging.getLogger(__name__)

    if not args.input_path.exists():
        logger.error("Input path does not exist: %s", args.input_path)
        print(f"Input path does not exist. See log: {args.log_file}")
        return 1

    model_path: Path | None = args.model_path if args.model_path.exists() else None
    if model_path is None:
        logger.warning(
            "Model path '%s' not found. Falling back to rule-based detection only.",
            args.model_path,
        )

    detector = AdaptiveHybridDetector(str(model_path) if model_path else None)

    if args.input_path.is_file():
        result = detector.process_single_file(args.input_path)
        if result is None:
            logger.error("Adaptive detection failed for %s", args.input_path)
            print(f"Detection failed for {args.input_path.name}. See log: {args.log_file}")
            return 2

        summary = summarize_result(result)

        # Always export machine-readable result for single files
        export_path = detector.export_machine_result(result, args.output_dir)
        logger.info(
            "Machine-readable export for %s saved to %s (%s)",
            args.input_path,
            export_path,
            summary,
        )

        if args.report:
            region_stats = {'S-D': 0, 'R-D': 0, '全文': 0}
            region_stats[result.learning_region] += 1
            detector.generate_batch_report([result], region_stats, args.output_dir)

            logger.info("Detection summary for %s: %s", args.input_path, summary)
            print(f"Detection completed for {args.input_path.name}. {summary}")
            print(f"Reports stored under: {args.output_dir.resolve()}")
        else:
            print(f"Machine-readable export written to: {export_path}")
            print(f"Detection summary: {summary}")

        print(f"Detailed log: {args.log_file}")
        return 0

    if args.input_path.is_dir():
        json_files = sorted(p for p in args.input_path.glob("*.json") if p.is_file())
        if args.max_files is not None:
            json_files = json_files[:args.max_files]

        if not json_files:
            logger.warning("No JSON files found under %s", args.input_path)
            print("No JSON files found to process.")
            print(f"Detailed log: {args.log_file}")
            return 0

        # Always process directory and export individual machine-readable files
        detector.process_sample_directory(args.input_path, args.output_dir, args.max_files, verbose=args.verbose)
        logger.info(
            "Batch detection finished for directory %s (max_files=%s).",
            args.input_path,
            args.max_files,
        )
        print(f"Batch detection completed for {args.input_path}.")
        print(f"Machine-readable exports stored under: {args.output_dir.resolve()}")

        if args.report:
            # Generate human-readable batch report
            json_files = sorted(p for p in args.input_path.glob("*.json") if p.is_file())
            if args.max_files is not None:
                json_files = json_files[:args.max_files]

            results = []
            region_stats = {'S-D': 0, 'R-D': 0, '全文': 0}

            for json_file in json_files:
                result = detector.process_single_file(json_file)
                if result:
                    results.append(result)
                    region_stats[result.learning_region] += 1

            if results:
                detector.generate_batch_report(results, region_stats, args.output_dir)
                logger.info("Human-readable batch report generated for %d files", len(results))
                print(f"Human-readable report stored under: {args.output_dir.resolve()}")

        print(f"Detailed log: {args.log_file}")
        return 0

    logger.error("Input path is neither a file nor a directory: %s", args.input_path)
    print(f"Unsupported input path. See log: {args.log_file}")
    return 1


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()