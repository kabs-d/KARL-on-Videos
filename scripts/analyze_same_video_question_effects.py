#!/usr/bin/env python3
"""Analyze same-video different-question Qwen/KARL outcome changes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON_TAGS = ("eps_003", "eps_005", "eps_007")


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0


def outcome_name(original_correct: bool, karl_correct: bool) -> str:
    if original_correct and karl_correct:
        return "kept_correct"
    if original_correct and not karl_correct:
        return "lost"
    if not original_correct and karl_correct:
        return "fixed"
    return "kept_wrong"


def short_question(text: str, limit: int = 96) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_rows(
    subset_rows: list[dict[str, Any]],
    original_by_uid: dict[str, dict[str, Any]],
    karl_by_eps: dict[str, dict[str, dict[str, Any]]],
    epsilon_tags: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    question_rows: list[dict[str, Any]] = []
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in subset_rows:
        uid = row["row_uid"]
        original = original_by_uid[uid]
        record = {
            "row_uid": uid,
            "video_id": row["video_id"],
            "question_id": row.get("question_id", ""),
            "question": row["question"],
            "primary_task_family": row.get("primary_task_family", ""),
            "family_memberships": row.get("family_memberships", []),
            "tags": row.get("tag", []),
            "answer_id": row.get("answer_id", ""),
            "correct_answer_label": original.get("correct_answer_label", ""),
            "original_prediction": original.get("parsed_prediction", ""),
            "original_correct": bool(original["is_correct"]),
        }
        by_video[row["video_id"]].append(record)

    video_summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for epsilon_tag in epsilon_tags:
        for video_id, records in sorted(by_video.items()):
            enriched_records: list[dict[str, Any]] = []
            for record in records:
                karl = karl_by_eps[epsilon_tag][record["row_uid"]]
                karl_correct = bool(karl["is_correct"])
                outcome = outcome_name(record["original_correct"], karl_correct)
                delta = int(karl_correct) - int(record["original_correct"])
                question_row = {
                    "epsilon_tag": epsilon_tag,
                    "video_id": video_id,
                    "row_uid": record["row_uid"],
                    "question_id": record["question_id"],
                    "primary_task_family": record["primary_task_family"],
                    "family_memberships": ";".join(record["family_memberships"]),
                    "tags": ";".join(record["tags"]),
                    "question": record["question"],
                    "correct_answer_label": record["correct_answer_label"],
                    "original_prediction": record["original_prediction"],
                    "karl_prediction": karl.get("parsed_prediction", ""),
                    "original_correct": int(record["original_correct"]),
                    "karl_correct": int(karl_correct),
                    "delta_correct": delta,
                    "outcome": outcome,
                    "active_tokens_mean": karl.get("active_tokens_mean", ""),
                    "reconstruction_l1_mean": karl.get("reconstruction_l1_mean", ""),
                }
                question_rows.append(question_row)
                enriched_records.append(question_row)

            families = sorted({row["primary_task_family"] for row in enriched_records})
            tags = sorted({tag for row in records for tag in row["tags"]})
            outcome_counts = Counter(row["outcome"] for row in enriched_records)
            fixed = outcome_counts["fixed"]
            lost = outcome_counts["lost"]
            original_acc = accuracy(enriched_records, "original_correct")
            karl_acc = accuracy(enriched_records, "karl_correct")
            summary_row = {
                "epsilon_tag": epsilon_tag,
                "video_id": video_id,
                "question_count": len(enriched_records),
                "family_count": len(families),
                "families": ";".join(families),
                "tag_count": len(tags),
                "tags": ";".join(tags),
                "original_accuracy": original_acc,
                "karl_accuracy": karl_acc,
                "accuracy_delta": karl_acc - original_acc,
                "fixed_count": fixed,
                "lost_count": lost,
                "kept_correct_count": outcome_counts["kept_correct"],
                "kept_wrong_count": outcome_counts["kept_wrong"],
                "mixed_delta": int(any(row["delta_correct"] > 0 for row in enriched_records) and any(row["delta_correct"] < 0 for row in enriched_records)),
                "any_changed": int(fixed > 0 or lost > 0),
            }
            video_summary_rows.append(summary_row)
            if fixed or lost:
                case_rows.append(summary_row)

    case_rows.sort(
        key=lambda row: (
            row["mixed_delta"],
            min(row["fixed_count"], row["lost_count"]),
            row["fixed_count"] + row["lost_count"],
            row["question_count"],
        ),
        reverse=True,
    )

    videos = list(by_video.values())
    overview = {
        "question_rows": len(subset_rows),
        "unique_videos": len(by_video),
        "epsilon_tags": list(epsilon_tags),
        "question_count_distribution": dict(sorted(Counter(len(rows) for rows in videos).items())),
        "family_count_distribution": dict(sorted(Counter(len({row["primary_task_family"] for row in rows}) for rows in videos).items())),
        "per_epsilon": {},
    }
    for epsilon_tag in epsilon_tags:
        rows = [row for row in video_summary_rows if row["epsilon_tag"] == epsilon_tag]
        overview["per_epsilon"][epsilon_tag] = {
            "videos": len(rows),
            "videos_any_changed": sum(row["any_changed"] for row in rows),
            "videos_mixed_fixed_and_lost": sum(row["mixed_delta"] for row in rows),
            "videos_with_fixed_question": sum(1 for row in rows if row["fixed_count"] > 0),
            "videos_with_lost_question": sum(1 for row in rows if row["lost_count"] > 0),
            "mean_original_accuracy_per_video": sum(row["original_accuracy"] for row in rows) / len(rows),
            "mean_karl_accuracy_per_video": sum(row["karl_accuracy"] for row in rows) / len(rows),
            "mean_accuracy_delta_per_video": sum(row["accuracy_delta"] for row in rows) / len(rows),
        }
    return question_rows, video_summary_rows, case_rows, overview


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_markdown(
    overview: dict[str, Any],
    video_summary_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    top_k: int,
) -> str:
    lines: list[str] = [
        "# Same-Video Different-Question Effects",
        "",
        "This analysis keeps visual evidence fixed at the video level and asks whether KARL reconstruction changes Qwen outcomes differently across questions about the same clip.",
        "",
        f"- question rows: {overview['question_rows']}",
        f"- unique videos: {overview['unique_videos']}",
        f"- epsilons: {', '.join(overview['epsilon_tags'])}",
        f"- question-count distribution per video: {overview['question_count_distribution']}",
        f"- task-family-count distribution per video: {overview['family_count_distribution']}",
        "",
        "## Per-Epsilon Summary",
        "",
    ]
    summary_rows = []
    for epsilon_tag in overview["epsilon_tags"]:
        row = overview["per_epsilon"][epsilon_tag]
        summary_rows.append(
            [
                epsilon_tag,
                row["videos"],
                row["videos_any_changed"],
                row["videos_with_fixed_question"],
                row["videos_with_lost_question"],
                row["videos_mixed_fixed_and_lost"],
                f"{row['mean_accuracy_delta_per_video']:.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["epsilon", "videos", "any changed", "with fixed", "with lost", "fixed+lost", "mean acc delta"],
            summary_rows,
        )
    )
    lines.extend(["", "## Strong Same-Video Cases", ""])
    for epsilon_tag in overview["epsilon_tags"]:
        cases = [
            row
            for row in video_summary_rows
            if row["epsilon_tag"] == epsilon_tag and row["fixed_count"] > 0 and row["lost_count"] > 0
        ]
        cases.sort(
            key=lambda row: (min(row["fixed_count"], row["lost_count"]), row["fixed_count"] + row["lost_count"], row["question_count"]),
            reverse=True,
        )
        lines.extend([f"### {epsilon_tag}", ""])
        case_table = [
            [
                row["video_id"],
                row["question_count"],
                row["family_count"],
                row["fixed_count"],
                row["lost_count"],
                f"{row['accuracy_delta']:.3f}",
                row["families"].replace(";", ", "),
            ]
            for row in cases[:top_k]
        ]
        if case_table:
            lines.extend(markdown_table(["video", "q", "families", "fixed", "lost", "acc delta", "family set"], case_table))
        else:
            lines.append("No videos had both fixed and lost questions.")
        lines.append("")

    lines.extend(["## Example Question-Level Changes", ""])
    eps007_cases = [
        row
        for row in video_summary_rows
        if row["epsilon_tag"] == "eps_007" and row["fixed_count"] > 0 and row["lost_count"] > 0
    ]
    eps007_cases.sort(
        key=lambda row: (min(row["fixed_count"], row["lost_count"]), row["fixed_count"] + row["lost_count"], row["question_count"]),
        reverse=True,
    )
    for case in eps007_cases[: min(5, top_k)]:
        lines.extend([f"### {case['video_id']} at eps_007", ""])
        rows = [
            row
            for row in question_rows
            if row["epsilon_tag"] == "eps_007" and row["video_id"] == case["video_id"] and row["outcome"] in {"fixed", "lost"}
        ]
        detail_rows = [
            [
                row["outcome"],
                row["primary_task_family"],
                row["tags"].replace(";", ", "),
                row["original_prediction"],
                row["karl_prediction"],
                short_question(row["question"]),
            ]
            for row in rows
        ]
        lines.extend(markdown_table(["outcome", "family", "tags", "orig", "karl", "question"], detail_rows))
        lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- The videos are shared, but the questions are not independent because several tags and families overlap.",
            "- A `fixed` question means original Qwen was wrong and Qwen on KARL reconstruction became correct.",
            "- A `lost` question means original Qwen was correct and Qwen on KARL reconstruction became wrong.",
            "- These examples support task-conditioned sensitivity to the same compressed visual evidence; they do not prove that KARL itself reasons about the question.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    base = workspace / "outputs" / "karl_mdl" / "same_video_task_control_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=Path,
        default=workspace / "outputs" / "curated_tasks" / "subsets" / "same_video_controls_seed17.jsonl",
    )
    parser.add_argument(
        "--original-predictions",
        type=Path,
        default=base / "qwen_original_baseline" / "predictions" / "qwen_uniform8_original.jsonl",
    )
    parser.add_argument(
        "--qwen-karl-prediction-dir",
        type=Path,
        default=base / "qwen_karl_reconstruction_baseline" / "predictions",
    )
    parser.add_argument("--epsilons", nargs="+", default=list(EPSILON_TAGS))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=base / "analysis_same_video_question_effects_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = workspace_root()
    subset_path = resolve_path(args.subset, workspace)
    original_path = resolve_path(args.original_predictions, workspace)
    prediction_dir = resolve_path(args.qwen_karl_prediction_dir, workspace)
    output_dir = resolve_path(args.output_dir, workspace)
    epsilon_tags = tuple(args.epsilons)

    subset_rows = read_jsonl(subset_path)
    original_by_uid = {row["row_uid"]: row for row in read_jsonl(original_path)}
    karl_by_eps = {
        epsilon_tag: {
            row["row_uid"]: row
            for row in read_jsonl(prediction_dir / f"qwen_uniform8_karl_{epsilon_tag}.jsonl")
        }
        for epsilon_tag in epsilon_tags
    }

    missing_original = sorted({row["row_uid"] for row in subset_rows} - set(original_by_uid))
    if missing_original:
        raise KeyError(f"Missing original predictions for {len(missing_original)} rows, e.g. {missing_original[:3]}")
    for epsilon_tag, rows_by_uid in karl_by_eps.items():
        missing = sorted({row["row_uid"] for row in subset_rows} - set(rows_by_uid))
        if missing:
            raise KeyError(f"Missing {epsilon_tag} KARL predictions for {len(missing)} rows, e.g. {missing[:3]}")

    question_rows, video_summary_rows, case_rows, overview = build_rows(
        subset_rows,
        original_by_uid,
        karl_by_eps,
        epsilon_tags,
    )

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    write_csv(
        tables_dir / "same_video_question_outcomes.csv",
        question_rows,
        [
            "epsilon_tag",
            "video_id",
            "row_uid",
            "question_id",
            "primary_task_family",
            "family_memberships",
            "tags",
            "question",
            "correct_answer_label",
            "original_prediction",
            "karl_prediction",
            "original_correct",
            "karl_correct",
            "delta_correct",
            "outcome",
            "active_tokens_mean",
            "reconstruction_l1_mean",
        ],
    )
    write_csv(
        tables_dir / "same_video_video_summary.csv",
        video_summary_rows,
        [
            "epsilon_tag",
            "video_id",
            "question_count",
            "family_count",
            "families",
            "tag_count",
            "tags",
            "original_accuracy",
            "karl_accuracy",
            "accuracy_delta",
            "fixed_count",
            "lost_count",
            "kept_correct_count",
            "kept_wrong_count",
            "mixed_delta",
            "any_changed",
        ],
    )
    write_csv(
        tables_dir / "same_video_case_index.csv",
        case_rows,
        [
            "epsilon_tag",
            "video_id",
            "question_count",
            "family_count",
            "families",
            "tag_count",
            "original_accuracy",
            "karl_accuracy",
            "accuracy_delta",
            "fixed_count",
            "lost_count",
            "kept_correct_count",
            "kept_wrong_count",
            "mixed_delta",
            "any_changed",
        ],
    )
    write_json(reports_dir / "same_video_question_effects_summary.json", overview)
    write_text(
        reports_dir / "same_video_question_effects_summary.md",
        build_markdown(overview, video_summary_rows, question_rows, args.top_k),
    )

    print(f"[same-video-effects] question rows: {len(question_rows)}")
    print(f"[same-video-effects] video summary rows: {len(video_summary_rows)}")
    print(f"[same-video-effects] report: {reports_dir / 'same_video_question_effects_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
