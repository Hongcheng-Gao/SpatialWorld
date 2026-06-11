#!/usr/bin/env python3
"""Aggregate benchmark CSV results by task category and complexity type.

Reads a results CSV produced by benchmark runners (columns: Task ID, Completed)
and joins it with ``task_classification_detail.csv`` to compute Task Success
Rate (TSR) per scenario category and per complexity type.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFICATION = REPO_ROOT / "task_classification_detail.csv"

CATEGORY_ORDER = ("Daily", "Work", "Entertain", "Travel", "Social")
TASK_TYPE_ORDER = ("Navigation", "Interaction", "Hybrid")


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").lstrip("\ufeff")


def _parse_completed(value: str | None) -> str | None:
    """Return 'success', 'failure', or None for pending/unknown."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan", "n/a", ""}:
        return None
    lowered = text.lower()
    if lowered in {"true", "1", "yes", "success", "pass", "passed"}:
        return "success"
    if lowered in {"false", "0", "no", "failure", "fail", "failed"}:
        return "failure"
    return None


def _find_column(fieldnames: list[str] | None, candidates: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None
    normalized = {_normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = _normalize_header(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_classification(path: Path) -> dict[str, dict[str, str]]:
    """Map task_id -> {environment, category, task_type, instruction}."""
    mapping: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")

        env_col = _find_column(reader.fieldnames, ("environment", "env"))
        task_col = _find_column(reader.fieldnames, ("task_id", "taskid", "task id"))
        category_col = _find_column(reader.fieldnames, ("category",))
        task_type_col = _find_column(reader.fieldnames, ("task_type", "complexity"))
        instruction_col = _find_column(reader.fieldnames, ("instruction",))

        if not task_col or not category_col:
            raise ValueError(
                f"Classification file must include task_id and category columns: {path}"
            )

        for row in reader:
            task_id = (row.get(task_col) or "").strip()
            if not task_id:
                continue
            mapping[task_id] = {
                "environment": (row.get(env_col) or "").strip() if env_col else "",
                "category": (row.get(category_col) or "").strip(),
                "task_type": (row.get(task_type_col) or "").strip() if task_type_col else "",
                "instruction": (row.get(instruction_col) or "").strip() if instruction_col else "",
            }
    return mapping


def load_results(path: Path) -> dict[str, str | None]:
    """Map task_id -> parsed completion status."""
    results: dict[str, str | None] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")

        task_col = _find_column(reader.fieldnames, ("task_id", "taskid", "task id"))
        completed_col = _find_column(reader.fieldnames, ("completed", "status", "result"))
        if not task_col or not completed_col:
            raise ValueError(
                f"Results file must include Task ID and Completed columns: {path}"
            )

        for row in reader:
            task_id = (row.get(task_col) or "").strip()
            if not task_id:
                continue
            results[task_id] = _parse_completed(row.get(completed_col))
    return results


def _ordered_keys(keys: set[str], preferred: tuple[str, ...]) -> list[str]:
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key not in preferred and key))
    return ordered


def aggregate(
    classification: dict[str, dict[str, str]],
    results: dict[str, str | None],
) -> dict:
    buckets: dict[str, dict[str, int]] = {
        "category": defaultdict(lambda: {"success": 0, "failure": 0, "pending": 0, "total": 0}),
        "task_type": defaultdict(lambda: {"success": 0, "failure": 0, "pending": 0, "total": 0}),
        "environment": defaultdict(lambda: {"success": 0, "failure": 0, "pending": 0, "total": 0}),
        "overall": {"success": 0, "failure": 0, "pending": 0, "total": 0},
    }
    missing_in_classification: list[str] = []
    missing_in_results: list[str] = []

    for task_id, meta in classification.items():
        status = results.get(task_id)
        if task_id not in results:
            missing_in_results.append(task_id)

        for bucket_name, bucket_key in (
            ("category", meta["category"] or "Unknown"),
            ("task_type", meta["task_type"] or "Unknown"),
            ("environment", meta["environment"] or "Unknown"),
        ):
            bucket = buckets[bucket_name][bucket_key]
            bucket["total"] += 1
            if status == "success":
                bucket["success"] += 1
            elif status == "failure":
                bucket["failure"] += 1
            else:
                bucket["pending"] += 1

        overall = buckets["overall"]
        overall["total"] += 1
        if status == "success":
            overall["success"] += 1
        elif status == "failure":
            overall["failure"] += 1
        else:
            overall["pending"] += 1

    for task_id in results:
        if task_id not in classification:
            missing_in_classification.append(task_id)

    return {
        "buckets": buckets,
        "missing_in_classification": sorted(missing_in_classification),
        "missing_in_results": sorted(missing_in_results),
    }


def _tsr(stats: dict[str, int]) -> float | None:
    evaluated = stats["success"] + stats["failure"]
    if evaluated == 0:
        return None
    return 100.0 * stats["success"] / evaluated


def _format_row(name: str, stats: dict[str, int]) -> list[str]:
    tsr = _tsr(stats)
    return [
        name,
        str(stats["total"]),
        str(stats["success"]),
        str(stats["failure"]),
        str(stats["pending"]),
        f"{tsr:.2f}" if tsr is not None else "n/a",
    ]


def print_summary(aggregated: dict, results_path: Path, classification_path: Path) -> None:
    buckets = aggregated["buckets"]
    print(f"Results file: {results_path}")
    print(f"Classification file: {classification_path}")
    print()

    headers = ["Group", "Total", "Success", "Failure", "Pending", "TSR (%)"]

    def print_table(title: str, data: dict[str, dict[str, int]], preferred: tuple[str, ...]) -> None:
        print(title)
        print(",".join(headers))
        for key in _ordered_keys(set(data), preferred):
            print(",".join(_format_row(key, data[key])))
        print()

    print_table("=== By Scenario Category ===", buckets["category"], CATEGORY_ORDER)
    print_table("=== By Complexity Type ===", buckets["task_type"], TASK_TYPE_ORDER)
    print_table("=== By Environment ===", buckets["environment"], ())

    overall = buckets["overall"]
    print("=== Overall ===")
    print(",".join(headers))
    print(",".join(_format_row("Overall", overall)))
    print()

    if aggregated["missing_in_classification"]:
        print(
            f"Warning: {len(aggregated['missing_in_classification'])} task IDs in results "
            "are missing from the classification file."
        )
    if aggregated["missing_in_results"]:
        print(
            f"Note: {len(aggregated['missing_in_results'])} classified tasks are missing "
            "from the results file (counted as pending)."
        )


def write_summary_csv(path: Path, aggregated: dict) -> None:
    buckets = aggregated["buckets"]
    rows: list[tuple[str, str, dict[str, int]]] = []
    for key in _ordered_keys(set(buckets["category"]), CATEGORY_ORDER):
        rows.append(("category", key, buckets["category"][key]))
    for key in _ordered_keys(set(buckets["task_type"]), TASK_TYPE_ORDER):
        rows.append(("task_type", key, buckets["task_type"][key]))
    for key in sorted(buckets["environment"]):
        rows.append(("environment", key, buckets["environment"][key]))
    rows.append(("overall", "Overall", buckets["overall"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["group_type", "group_name", "total", "success", "failure", "pending", "tsr_percent"]
        )
        for group_type, group_name, stats in rows:
            tsr = _tsr(stats)
            writer.writerow(
                [
                    group_type,
                    group_name,
                    stats["total"],
                    stats["success"],
                    stats["failure"],
                    stats["pending"],
                    f"{tsr:.4f}" if tsr is not None else "",
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark CSV results by task category."
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Path to a benchmark results CSV (Task ID, Completed).",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=DEFAULT_CLASSIFICATION,
        help=f"Path to task classification CSV (default: {DEFAULT_CLASSIFICATION.name}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregated summary as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results.is_file():
        print(f"Results file not found: {args.results}", file=sys.stderr)
        return 1
    if not args.classification.is_file():
        print(f"Classification file not found: {args.classification}", file=sys.stderr)
        return 1

    classification = load_classification(args.classification)
    results = load_results(args.results)
    aggregated = aggregate(classification, results)

    print_summary(aggregated, args.results, args.classification)
    if args.output:
        write_summary_csv(args.output, aggregated)
        print(f"Wrote summary CSV to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
