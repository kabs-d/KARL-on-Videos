#!/usr/bin/env python3
"""Build Perception Test MCQ manifests from extracted annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


AREA_VALUES = {"Memory", "Abstraction", "Physics", "Semantics"}
REASONING_VALUES = {"Descriptive", "Explanatory", "Predictive", "Counterfactual"}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def looks_like_video_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("video_")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_json_files(path: Path) -> Iterator[Path]:
    if not path.exists():
        return
    yield from sorted(path.rglob("*.json"))


def iter_video_files(path: Path) -> Iterator[Path]:
    if not path.exists():
        return
    for suffix in ("*.mp4", "*.MP4"):
        yield from sorted(path.rglob(suffix))


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def normalize_answer_id(value: Any) -> int | str | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return None


def answer_id_is_valid(answer_id: int | str | None, options: list[Any]) -> bool:
    if isinstance(answer_id, int):
        return 0 <= answer_id < len(options)
    return isinstance(answer_id, str) and answer_id in {str(option) for option in options}


def extract_options(raw_options: Any) -> list[str]:
    if isinstance(raw_options, list):
        return [str(option) for option in raw_options]
    if isinstance(raw_options, dict):
        return [str(raw_options[key]) for key in sorted(raw_options)]
    return []


def find_video_id(record: dict[str, Any], context_video_id: str | None) -> str | None:
    for key in ("video_id", "videoId", "video", "clip_id"):
        value = record.get(key)
        if looks_like_video_id(value):
            return value
    return context_video_id


def is_mcq_record(record: dict[str, Any]) -> bool:
    return "question" in record and "options" in record and "answer_id" in record


def traverse_annotations(
    node: Any,
    *,
    source: Path,
    context_video_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from traverse_annotations(item, source=source, context_video_id=context_video_id)
        return

    if not isinstance(node, dict):
        return

    node_video_id = find_video_id(node, context_video_id)
    if is_mcq_record(node):
        video_id = node_video_id
        if video_id is not None:
            options = extract_options(node.get("options"))
            answer_id = normalize_answer_id(node.get("answer_id"))
            yield {
                "video_id": video_id,
                "question_id": str(node.get("id", node.get("question_id", ""))),
                "question": str(node.get("question", "")),
                "options": options,
                "answer_id": answer_id,
                "area": str(node.get("area", "")),
                "reasoning": str(node.get("reasoning", "")),
                "tag": normalize_tags(node.get("tag")),
                "source_annotation": str(source),
            }

    for key, value in node.items():
        child_context = node_video_id
        if looks_like_video_id(key):
            child_context = key
        yield from traverse_annotations(value, source=source, context_video_id=child_context)


def load_challenge_ids(workspace: Path, split: str) -> set[str] | None:
    if split != "train":
        return None
    path = workspace / "data" / "challenge_ids" / "mc_question_train_id_list.csv"
    if not path.exists():
        return None

    video_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if looks_like_video_id(cell):
                    video_ids.add(cell)
    return video_ids


def load_cut_mapping(workspace: Path, split: str) -> dict[str, int]:
    if split != "train":
        return {}
    path = workspace / "data" / "cut_mappings" / "cut_frame_mapping_train.json"
    if not path.exists():
        return {}
    data = load_json(path)
    mapping: dict[str, int] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if looks_like_video_id(key):
                try:
                    mapping[key] = int(value)
                except (TypeError, ValueError):
                    continue
    return mapping


def build_video_index(workspace: Path, split: str) -> dict[str, Path]:
    root = workspace / "data" / "videos" / split
    index: dict[str, Path] = {}
    for path in iter_video_files(root):
        index.setdefault(path.stem, path)
    return index


def annotation_roots(workspace: Path, split: str) -> list[Path]:
    root = workspace / "data" / "annotations" / split
    if split == "train":
        return [root / "mcq", root / "full", root]
    return [root]


def build_manifest(workspace: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_index = build_video_index(workspace, split)
    challenge_ids = load_challenge_ids(workspace, split)
    cut_mapping = load_cut_mapping(workspace, split)
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "raw_records": 0,
        "filtered_by_challenge_ids": 0,
        "missing_video_path": 0,
        "invalid_option_count": 0,
        "invalid_answer_id": 0,
        "annotation_files": [],
    }

    seen_json: set[Path] = set()
    for root in annotation_roots(workspace, split):
        for json_path in iter_json_files(root):
            if json_path in seen_json:
                continue
            seen_json.add(json_path)
            stats["annotation_files"].append(str(json_path.relative_to(workspace)))
            try:
                data = load_json(json_path)
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping invalid JSON {json_path}: {exc}", file=sys.stderr)
                continue

            for record in traverse_annotations(data, source=json_path.relative_to(workspace)):
                stats["raw_records"] += 1
                video_id = record["video_id"]
                if challenge_ids is not None and video_id not in challenge_ids:
                    stats["filtered_by_challenge_ids"] += 1
                    continue
                if len(record["options"]) != 3:
                    stats["invalid_option_count"] += 1
                    continue
                if not answer_id_is_valid(record["answer_id"], record["options"]):
                    stats["invalid_answer_id"] += 1
                    continue

                video_path = video_index.get(video_id)
                if video_path is None:
                    stats["missing_video_path"] += 1
                relative_video_path = str(video_path.relative_to(workspace)) if video_path else ""
                question_id = record["question_id"] or f"{video_id}_{stats['raw_records']}"
                row = {
                    "split": split,
                    "video_id": video_id,
                    "video_path": relative_video_path,
                    "question_id": question_id,
                    "question": record["question"],
                    "options": record["options"],
                    "answer_id": record["answer_id"],
                    "area": record["area"],
                    "reasoning": record["reasoning"],
                    "tag": record["tag"],
                    "cut_frame": cut_mapping.get(video_id),
                    "source_annotation": record["source_annotation"],
                }
                rows_by_key.setdefault((split, video_id, question_id), row)

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    return rows, stats


def write_manifest(workspace: Path, split: str, rows: list[dict[str, Any]]) -> Path:
    output = workspace / "manifests" / f"{split}_mcq.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument("--splits", nargs="+", choices=("sample", "train"), default=("sample", "train"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    for split in args.splits:
        rows, stats = build_manifest(workspace, split)
        output = write_manifest(workspace, split, rows)
        print(f"[manifest] {split}: wrote {len(rows)} rows to {output}")
        print(
            "[manifest] "
            f"raw={stats['raw_records']} "
            f"filtered={stats['filtered_by_challenge_ids']} "
            f"missing_videos={stats['missing_video_path']} "
            f"bad_options={stats['invalid_option_count']} "
            f"bad_answers={stats['invalid_answer_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
