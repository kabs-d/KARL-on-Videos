#!/usr/bin/env python3
"""Curate concrete Perception Test task families for KARL-QA experiments."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 17

PRIMARY_PRIORITY: tuple[tuple[str, set[str]], ...] = (
    ("occlusion_permanence", {"occlusion", "object permanence", "containment", "solidity", "collision"}),
    ("object_counting", {"object counting"}),
    ("spatial_relations", {"spatial relations", "part recognition"}),
    ("temporal_event", {"motion", "sequencing", "task completion", "action counting", "event counting", "event recall"}),
    ("recognition_control", {"object recognition", "place recognition", "action recognition"}),
)

FAMILY_TAGS: dict[str, set[str]] = dict(PRIMARY_PRIORITY)
FAMILY_ORDER: tuple[str, ...] = tuple(name for name, _tags in PRIMARY_PRIORITY)
HARD_FAMILIES = {"occlusion_permanence", "object_counting", "spatial_relations", "temporal_event"}
RESPONSIBILITY_FAMILIES = {"occlusion_permanence", "object_counting", "spatial_relations", "recognition_control", "temporal_event"}
ABLATION_FAMILIES = {"occlusion_permanence", "object_counting", "spatial_relations", "recognition_control"}
OBJECT_SPECIFIC_RE = re.compile(
    r"\b(object|objects|hidden|where|which|what|how many|count|left|right|under|behind|holding|using|person)\b",
    re.IGNORECASE,
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_tags(row: dict[str, Any]) -> set[str]:
    tags = row.get("tag", [])
    if isinstance(tags, list):
        return {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    if tags is None:
        return set()
    return {str(tags).strip().lower()}


def family_memberships(tags: set[str]) -> list[str]:
    memberships = [family for family in FAMILY_ORDER if tags & FAMILY_TAGS[family]]
    return memberships


def primary_family(tags: set[str]) -> str:
    for family, family_tags in PRIMARY_PRIORITY:
        if tags & family_tags:
            return family
    return "other"


def row_uid(row: dict[str, Any]) -> str:
    return f"{row.get('split', 'train')}:{row['video_id']}:{row['question_id']}"


def enrich_rows(rows: list[dict[str, Any]], workspace: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    skipped_missing_video = 0
    skipped_bad_options = 0
    for row in rows:
        video_path = workspace / str(row.get("video_path", ""))
        if not row.get("video_path") or not video_path.exists():
            skipped_missing_video += 1
            continue
        if not isinstance(row.get("options"), list) or len(row["options"]) != 3:
            skipped_bad_options += 1
            continue

        tags = normalize_tags(row)
        memberships = family_memberships(tags)
        primary = primary_family(tags)
        new_row = dict(row)
        new_row["row_uid"] = row_uid(row)
        new_row["primary_task_family"] = primary
        new_row["family_memberships"] = memberships
        new_row["is_concrete_visual"] = primary != "other"
        new_row["is_temporal_candidate"] = primary in {"temporal_event", "occlusion_permanence"} or "temporal_event" in memberships
        new_row["is_token_responsibility_candidate"] = primary in RESPONSIBILITY_FAMILIES
        new_row["is_task_critical_ablation_candidate"] = (
            primary in ABLATION_FAMILIES and bool(OBJECT_SPECIFIC_RE.search(str(row.get("question", ""))))
        )
        enriched.append(new_row)

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_video[row["video_id"]].append(row)

    for video_id, video_rows in by_video.items():
        video_families = sorted({row["primary_task_family"] for row in video_rows if row["primary_task_family"] != "other"})
        for row in video_rows:
            row["same_video_task_families"] = video_families
            row["same_video_family_count"] = len(video_families)
            row["is_same_video_candidate"] = len(video_families) >= 2

    stats = {
        "input_rows": len(rows),
        "skipped_missing_video": skipped_missing_video,
        "skipped_bad_options": skipped_bad_options,
        "enriched_rows": len(enriched),
    }
    return enriched, stats


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output = sorted(rows, key=lambda row: row["row_uid"])
    rng.shuffle(output)
    return output


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    exclude_uids: set[str] | None = None,
    prefer_same_video: bool = False,
    one_per_video_first: bool = True,
) -> list[dict[str, Any]]:
    exclude_uids = exclude_uids or set()
    candidates = [row for row in rows if row["row_uid"] not in exclude_uids]
    if prefer_same_video:
        preferred = [row for row in candidates if row.get("is_same_video_candidate")]
        fallback = [row for row in candidates if not row.get("is_same_video_candidate")]
        candidates = shuffled(preferred, seed) + shuffled(fallback, seed + 1009)
    else:
        candidates = shuffled(candidates, seed)

    selected: list[dict[str, Any]] = []
    selected_videos: set[str] = set()
    if one_per_video_first:
        for row in candidates:
            if row["video_id"] in selected_videos:
                continue
            selected.append(row)
            selected_videos.add(row["video_id"])
            if len(selected) >= count:
                return selected

    selected_uids = {row["row_uid"] for row in selected}
    for row in candidates:
        if row["row_uid"] in selected_uids:
            continue
        selected.append(row)
        selected_uids.add(row["row_uid"])
        if len(selected) >= count:
            break
    return selected


def balanced_subset(
    rows_by_family: dict[str, list[dict[str, Any]]],
    per_family: int,
    *,
    seed: int,
    exclude_uids: set[str] | None = None,
    prefer_same_video: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for offset, family in enumerate(FAMILY_ORDER):
        family_rows = rows_by_family[family]
        family_selected = sample_rows(
            family_rows,
            per_family,
            seed=seed + offset * 13,
            exclude_uids=exclude_uids,
            prefer_same_video=prefer_same_video,
        )
        selected.extend(family_selected)
    return sorted(selected, key=lambda row: (row["primary_task_family"], row["video_id"], row["question_id"]))


def build_same_video_controls(curated_rows: list[dict[str, Any]], *, seed: int, max_videos: int = 60) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curated_rows:
        if row.get("is_same_video_candidate"):
            by_video[row["video_id"]].append(row)

    def score(video_rows: list[dict[str, Any]]) -> tuple[int, int, int, str]:
        families = {row["primary_task_family"] for row in video_rows}
        has_recognition_control = int("recognition_control" in families)
        hard_count = len(families & HARD_FAMILIES)
        question_count = len(video_rows)
        return (has_recognition_control, hard_count, question_count, video_rows[0]["video_id"])

    rng = random.Random(seed)
    video_groups = list(by_video.values())
    rng.shuffle(video_groups)
    video_groups.sort(key=score, reverse=True)
    selected_videos = {rows[0]["video_id"] for rows in video_groups[:max_videos]}
    selected = [row for row in curated_rows if row["video_id"] in selected_videos]
    return sorted(selected, key=lambda row: (row["video_id"], row["primary_task_family"], row["question_id"]))


def build_subset_banks(curated_rows: list[dict[str, Any]], rows_by_family: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, list[dict[str, Any]]]:
    calibration = balanced_subset(rows_by_family, 30, seed=seed, prefer_same_video=True)
    calibration_uids = {row["row_uid"] for row in calibration}
    main = balanced_subset(rows_by_family, 60, seed=seed + 101, exclude_uids=calibration_uids, prefer_same_video=True)
    same_video = build_same_video_controls(curated_rows, seed=seed + 202, max_videos=60)

    temporal_rows = []
    temporal_specs = (
        ("temporal_event", 20),
        ("occlusion_permanence", 20),
        ("recognition_control", 20),
    )
    for offset, (family, count) in enumerate(temporal_specs):
        temporal_rows.extend(
            sample_rows(
                [row for row in rows_by_family[family] if row.get("is_temporal_candidate") or family == "recognition_control"],
                count,
                seed=seed + 303 + offset * 7,
                prefer_same_video=True,
            )
        )

    responsibility_rows = []
    for offset, family in enumerate(FAMILY_ORDER):
        candidates = [row for row in rows_by_family[family] if row.get("is_token_responsibility_candidate")]
        responsibility_rows.extend(sample_rows(candidates, 8, seed=seed + 404 + offset * 7, prefer_same_video=True))

    ablation_rows = []
    for offset, family in enumerate(("occlusion_permanence", "object_counting", "spatial_relations", "recognition_control")):
        candidates = [row for row in rows_by_family[family] if row.get("is_task_critical_ablation_candidate")]
        ablation_rows.extend(sample_rows(candidates, 10, seed=seed + 505 + offset * 7, prefer_same_video=True))

    return {
        "calibration_balanced_150_seed17": sorted(calibration, key=lambda row: row["row_uid"]),
        "main_balanced_300_seed17": sorted(main, key=lambda row: row["row_uid"]),
        "same_video_controls_seed17": sorted(same_video, key=lambda row: row["row_uid"]),
        "temporal_signature_bank_seed17": sorted(temporal_rows, key=lambda row: row["row_uid"]),
        "token_responsibility_bank_seed17": sorted(responsibility_rows, key=lambda row: row["row_uid"]),
        "task_critical_ablation_bank_seed17": sorted(ablation_rows, key=lambda row: row["row_uid"]),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def membership_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("family_memberships", []))
    return dict(sorted(counts.items()))


def make_summary(
    all_rows: list[dict[str, Any]],
    curated_rows: list[dict[str, Any]],
    subsets: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    primary_counts = count_by(curated_rows, "primary_task_family")
    video_counts_by_family = {
        family: len({row["video_id"] for row in curated_rows if row["primary_task_family"] == family})
        for family in FAMILY_ORDER
    }
    subset_summary = {}
    for name, rows in subsets.items():
        subset_summary[name] = {
            "rows": len(rows),
            "unique_videos": len({row["video_id"] for row in rows}),
            "primary_task_family": count_by(rows, "primary_task_family"),
            "area": count_by(rows, "area"),
            "reasoning": count_by(rows, "reasoning"),
        }

    calibration_uids = {row["row_uid"] for row in subsets["calibration_balanced_150_seed17"]}
    main_uids = {row["row_uid"] for row in subsets["main_balanced_300_seed17"]}
    return {
        **stats,
        "source_train_rows": len(all_rows),
        "curated_rows": len(curated_rows),
        "excluded_other_rows": sum(1 for row in all_rows if row.get("primary_task_family") == "other"),
        "curated_unique_videos": len({row["video_id"] for row in curated_rows}),
        "primary_family_counts": primary_counts,
        "membership_counts": membership_counts(curated_rows),
        "video_counts_by_primary_family": video_counts_by_family,
        "same_video_candidate_rows": sum(1 for row in curated_rows if row.get("is_same_video_candidate")),
        "same_video_candidate_videos": len({row["video_id"] for row in curated_rows if row.get("is_same_video_candidate")}),
        "temporal_candidate_rows": sum(1 for row in curated_rows if row.get("is_temporal_candidate")),
        "token_responsibility_candidate_rows": sum(1 for row in curated_rows if row.get("is_token_responsibility_candidate")),
        "task_critical_ablation_candidate_rows": sum(1 for row in curated_rows if row.get("is_task_critical_ablation_candidate")),
        "calibration_main_overlap_rows": len(calibration_uids & main_uids),
        "subsets": subset_summary,
    }


def markdown_table(counter: dict[str, int]) -> list[str]:
    return [f"- {key}: {value}" for key, value in counter.items()]


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Curated Perception Test Task Data",
        "",
        "Scope: train MCQ rows assigned to concrete KARL-QA task families.",
        "",
        "## Overview",
        "",
        f"- source train rows: {summary['source_train_rows']}",
        f"- curated rows: {summary['curated_rows']}",
        f"- excluded other rows: {summary['excluded_other_rows']}",
        f"- curated unique videos: {summary['curated_unique_videos']}",
        f"- same-video candidate videos: {summary['same_video_candidate_videos']}",
        f"- calibration/main overlap rows: {summary['calibration_main_overlap_rows']}",
        "",
        "## Primary Family Counts",
        "",
        *markdown_table(summary["primary_family_counts"]),
        "",
        "## Multi-Label Membership Counts",
        "",
        *markdown_table(summary["membership_counts"]),
        "",
        "## Subsets",
        "",
    ]
    for name, info in summary["subsets"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- rows: {info['rows']}",
                f"- unique videos: {info['unique_videos']}",
                "- primary family:",
                *[f"  - {key}: {value}" for key, value in info["primary_task_family"].items()],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    source = workspace / "manifests" / "train_mcq.jsonl"
    output_root = workspace / "outputs" / "curated_tasks"
    manifests_dir = output_root / "manifests"
    subsets_dir = output_root / "subsets"
    reports_dir = output_root / "reports"

    raw_rows = read_jsonl(source)
    enriched_rows, stats = enrich_rows(raw_rows, workspace)
    curated_rows = [row for row in enriched_rows if row["primary_task_family"] != "other"]
    curated_rows = sorted(curated_rows, key=lambda row: row["row_uid"])
    rows_by_family = {
        family: [row for row in curated_rows if row["primary_task_family"] == family]
        for family in FAMILY_ORDER
    }

    write_jsonl(manifests_dir / "curated_master.jsonl", curated_rows)
    for family in FAMILY_ORDER:
        write_jsonl(manifests_dir / f"family_{family}.jsonl", rows_by_family[family])

    subsets = build_subset_banks(curated_rows, rows_by_family, seed=args.seed)
    for name, rows in subsets.items():
        write_jsonl(subsets_dir / f"{name}.jsonl", rows)

    summary = make_summary(enriched_rows, curated_rows, subsets, stats)
    write_json(reports_dir / "curated_task_data_summary.json", summary)
    write_summary_markdown(reports_dir / "curated_task_data_summary.md", summary)

    print(f"[curate] source rows: {summary['source_train_rows']}")
    print(f"[curate] curated rows: {summary['curated_rows']}")
    print(f"[curate] curated unique videos: {summary['curated_unique_videos']}")
    for family in FAMILY_ORDER:
        print(f"[curate] {family}: {summary['primary_family_counts'].get(family, 0)} rows")
    for name, info in summary["subsets"].items():
        print(f"[curate] subset {name}: {info['rows']} rows | {info['unique_videos']} videos")
    print(f"[curate] wrote {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
