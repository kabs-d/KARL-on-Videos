#!/usr/bin/env python3
"""Combined Qwen/KARL reconstruction tradeoff analysis over curated PT subsets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON_TAGS = ("eps_003", "eps_005", "eps_007")
MAJOR_TAGS: tuple[str, ...] = (
    "spatial relations",
    "motion",
    "object recognition",
    "place recognition",
    "solidity",
    "sequencing",
    "occlusion",
    "object permanence",
    "object counting",
    "collision",
    "action counting",
    "part recognition",
)
CONDITION_ORDER = ("original", "eps_003", "eps_005", "eps_007")


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def tags_of(row: dict[str, Any]) -> set[str]:
    tags = row.get("tag") or row.get("tags") or []
    if isinstance(tags, str):
        return {tag.strip().lower() for tag in tags.split(";") if tag.strip()}
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def accuracy(rows: list[dict[str, Any]], key: str = "is_correct") -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["row_uid"]: row for row in read_jsonl(path)}


def load_karl_prediction_maps(prediction_dir: Path, epsilon_tags: tuple[str, ...]) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for epsilon_tag in epsilon_tags:
        path = prediction_dir / f"qwen_uniform8_karl_{epsilon_tag}.jsonl"
        out[epsilon_tag] = load_prediction_map(path)
    return out


def add_subset_rows(
    rows_by_uid: dict[str, dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    subset_name: str,
) -> None:
    for row in subset_rows:
        uid = row["row_uid"]
        if uid not in rows_by_uid:
            clean = dict(row)
            clean["source_subsets"] = []
            rows_by_uid[uid] = clean
        if subset_name not in rows_by_uid[uid]["source_subsets"]:
            rows_by_uid[uid]["source_subsets"].append(subset_name)


def source_for_row(row: dict[str, Any]) -> str:
    # Prefer same-video predictions for exact overlaps so the same-video case study
    # and the combined table share the same rows.
    subsets = row.get("source_subsets", [])
    if "same_video_controls_seed17" in subsets:
        return "same_video_controls_seed17"
    return "main_balanced_300_seed17"


def build_question_rows(
    combined_rows: list[dict[str, Any]],
    original_by_source: dict[str, dict[str, dict[str, Any]]],
    karl_by_source: dict[str, dict[str, dict[str, dict[str, Any]]]],
    epsilon_tags: tuple[str, ...],
) -> list[dict[str, Any]]:
    question_rows: list[dict[str, Any]] = []
    for row in combined_rows:
        uid = row["row_uid"]
        source = source_for_row(row)
        original = original_by_source[source][uid]
        base = {
            "row_uid": uid,
            "video_id": row["video_id"],
            "source_subsets": ";".join(sorted(row.get("source_subsets", []))),
            "prediction_source": source,
            "primary_task_family": row.get("primary_task_family", ""),
            "family_memberships": ";".join(row.get("family_memberships", [])),
            "tags": ";".join(sorted(tags_of(row))),
            "area": row.get("area", ""),
            "reasoning": row.get("reasoning", ""),
            "question": row.get("question", ""),
            "correct_answer_label": original.get("correct_answer_label", ""),
            "original_prediction": original.get("parsed_prediction", ""),
            "original_correct": int(bool(original["is_correct"])),
        }
        question_rows.append(
            {
                **base,
                "condition": "original",
                "epsilon_tag": "original",
                "karl_prediction": "",
                "karl_correct": "",
                "is_correct": int(bool(original["is_correct"])),
                "delta_from_original": 0,
                "outcome": "original",
                "active_tokens_mean": 256.0,
                "reconstruction_l1_mean": 0.0,
            }
        )
        for epsilon_tag in epsilon_tags:
            karl = karl_by_source[source][epsilon_tag][uid]
            karl_correct = bool(karl["is_correct"])
            original_correct = bool(original["is_correct"])
            if original_correct and karl_correct:
                outcome = "kept_correct"
            elif original_correct and not karl_correct:
                outcome = "lost"
            elif not original_correct and karl_correct:
                outcome = "fixed"
            else:
                outcome = "kept_wrong"
            question_rows.append(
                {
                    **base,
                    "condition": epsilon_tag,
                    "epsilon_tag": epsilon_tag,
                    "karl_prediction": karl.get("parsed_prediction", ""),
                    "karl_correct": int(karl_correct),
                    "is_correct": int(karl_correct),
                    "delta_from_original": int(karl_correct) - int(original_correct),
                    "outcome": outcome,
                    "active_tokens_mean": float(karl.get("active_tokens_mean", 0.0)),
                    "reconstruction_l1_mean": float(karl.get("reconstruction_l1_mean", 0.0)),
                }
            )
    return question_rows


def summarize_group(rows: list[dict[str, Any]], label: str, value: str, condition: str) -> dict[str, Any]:
    cond_rows = [row for row in rows if row["condition"] == condition]
    original_rows = [row for row in rows if row["condition"] == "original"]
    original_by_uid = {row["row_uid"]: row for row in original_rows}
    summary = {
        label: value,
        "condition": condition,
        "rows": len(cond_rows),
        "accuracy": accuracy(cond_rows),
        "active_tokens_mean": mean([float(row["active_tokens_mean"]) for row in cond_rows]),
        "reconstruction_l1_mean": mean([float(row["reconstruction_l1_mean"]) for row in cond_rows]),
        "fixed_count": 0,
        "lost_count": 0,
        "kept_correct_count": 0,
        "kept_wrong_count": 0,
    }
    if condition != "original":
        outcomes = Counter(row["outcome"] for row in cond_rows)
        summary.update(
            {
                "original_accuracy": accuracy([original_by_uid[row["row_uid"]] for row in cond_rows]),
                "accuracy_delta_from_original": accuracy(cond_rows) - accuracy([original_by_uid[row["row_uid"]] for row in cond_rows]),
                "fixed_count": outcomes["fixed"],
                "lost_count": outcomes["lost"],
                "kept_correct_count": outcomes["kept_correct"],
                "kept_wrong_count": outcomes["kept_wrong"],
            }
        )
    else:
        summary["original_accuracy"] = summary["accuracy"]
        summary["accuracy_delta_from_original"] = 0.0
    return summary


def build_summaries(question_rows: list[dict[str, Any]], construction: dict[str, Any]) -> dict[str, Any]:
    base_rows = [row for row in question_rows if row["condition"] == "original"]
    condition_summary = {
        condition: summarize_group(question_rows, "scope", "combined", condition)
        for condition in CONDITION_ORDER
    }

    family_summaries: list[dict[str, Any]] = []
    families = sorted({row["primary_task_family"] for row in base_rows})
    for family in families:
        group_rows = [row for row in question_rows if row["primary_task_family"] == family]
        for condition in CONDITION_ORDER:
            family_summaries.append(summarize_group(group_rows, "family", family, condition))

    tag_summaries: list[dict[str, Any]] = []
    for tag in MAJOR_TAGS:
        group_rows = [row for row in question_rows if tag in set(row["tags"].split(";"))]
        if not group_rows:
            continue
        for condition in CONDITION_ORDER:
            tag_summaries.append(summarize_group(group_rows, "tag", tag, condition))

    source_summaries: list[dict[str, Any]] = []
    for source_value in sorted({row["source_subsets"] for row in base_rows}):
        group_rows = [row for row in question_rows if row["source_subsets"] == source_value]
        for condition in CONDITION_ORDER:
            source_summaries.append(summarize_group(group_rows, "source_subsets", source_value, condition))

    return {
        "dataset_construction": construction,
        "condition_summary": condition_summary,
        "family_summaries": family_summaries,
        "tag_summaries": tag_summaries,
        "source_summaries": source_summaries,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    construction = summary["dataset_construction"]
    lines = [
        "# Combined Qwen/KARL Reconstruction Tradeoff",
        "",
        "This is the primary overall analysis for the Qwen side of the KARL extension.",
        "",
        "## Dataset Construction",
        "",
        "The dataset is derived from the Perception Test train MCQ manifest.",
        "",
        f"- source train MCQ rows: {construction['source_train_rows']}",
        f"- rows with valid local videos and 3 MCQ options: {construction['enriched_rows']}",
        f"- concrete curated rows after task-family assignment: {construction['curated_rows']}",
        f"- curated unique videos: {construction['curated_unique_videos']}",
        f"- main balanced subset: {construction['main_rows']} rows, {construction['main_unique_videos']} videos",
        f"- same-video control subset: {construction['same_video_rows']} rows, {construction['same_video_unique_videos']} videos",
        f"- exact row overlap between subsets: {construction['overlap_rows']}",
        f"- combined deduplicated analysis set: {construction['combined_rows']} rows, {construction['combined_unique_videos']} videos",
        "",
        "Task families were assigned from Perception Test tags using the curation script, then the combined analysis deduplicated exact `row_uid` overlaps.",
        "",
        "## Global Tradeoff",
        "",
    ]
    global_rows = []
    for condition in CONDITION_ORDER:
        row = summary["condition_summary"][condition]
        global_rows.append(
            [
                condition,
                row["rows"],
                fmt(row["accuracy"]),
                fmt(row["accuracy_delta_from_original"]),
                fmt(row["active_tokens_mean"], 2),
                fmt(row["reconstruction_l1_mean"], 5),
                row["fixed_count"],
                row["lost_count"],
            ]
        )
    lines.extend(
        markdown_table(
            ["condition", "rows", "accuracy", "delta", "active mean", "L1 mean", "fixed", "lost"],
            global_rows,
        )
    )

    lines.extend(["", "## By Primary Task Family", ""])
    family_rows = []
    by_family_condition = {
        (row["family"], row["condition"]): row
        for row in summary["family_summaries"]
    }
    for family in sorted({row["family"] for row in summary["family_summaries"]}):
        original = by_family_condition[(family, "original")]
        eps003 = by_family_condition[(family, "eps_003")]
        eps005 = by_family_condition[(family, "eps_005")]
        eps007 = by_family_condition[(family, "eps_007")]
        family_rows.append(
            [
                family,
                original["rows"],
                fmt(original["accuracy"]),
                fmt(eps003["accuracy"]),
                fmt(eps005["accuracy"]),
                fmt(eps007["accuracy"]),
                fmt(eps007["accuracy_delta_from_original"]),
            ]
        )
    lines.extend(markdown_table(["family", "n", "original", "eps003", "eps005", "eps007", "eps007 delta"], family_rows))

    lines.extend(["", "## By Major Tag", ""])
    by_tag_condition = {
        (row["tag"], row["condition"]): row
        for row in summary["tag_summaries"]
    }
    tag_rows = []
    for tag in MAJOR_TAGS:
        if (tag, "original") not in by_tag_condition:
            continue
        original = by_tag_condition[(tag, "original")]
        eps003 = by_tag_condition[(tag, "eps_003")]
        eps005 = by_tag_condition[(tag, "eps_005")]
        eps007 = by_tag_condition[(tag, "eps_007")]
        tag_rows.append(
            [
                tag,
                original["rows"],
                fmt(original["accuracy"]),
                fmt(eps003["accuracy"]),
                fmt(eps005["accuracy"]),
                fmt(eps007["accuracy"]),
                fmt(eps007["accuracy_delta_from_original"]),
            ]
        )
    lines.extend(markdown_table(["tag", "n", "original", "eps003", "eps005", "eps007", "eps007 delta"], tag_rows))

    hurt = sorted(
        [
            row
            for row in summary["tag_summaries"]
            if row.get("condition") == "eps_007"
        ],
        key=lambda row: row["accuracy_delta_from_original"],
    )
    lines.extend(["", "## Most Compression-Sensitive Tags At eps007", ""])
    lines.extend(
        markdown_table(
            ["tag", "n", "original", "eps007", "delta"],
            [
                [
                    row["tag"],
                    row["rows"],
                    fmt(row["original_accuracy"]),
                    fmt(row["accuracy"]),
                    fmt(row["accuracy_delta_from_original"]),
                ]
                for row in hurt
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This combined table is the headline Qwen/KARL result.",
            "- The same-video subset is intentionally overrepresented in this union because it provides controlled multi-question videos.",
            "- Exact overlapping questions are counted once.",
            "- The same-video report should be used for case studies; this combined report should be used for overall accuracy and tag/family trends.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary_tables(output_dir: Path, question_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    tables_dir = output_dir / "tables"
    write_csv(
        tables_dir / "combined_question_outcomes.csv",
        question_rows,
        [
            "condition",
            "epsilon_tag",
            "row_uid",
            "video_id",
            "source_subsets",
            "prediction_source",
            "primary_task_family",
            "family_memberships",
            "tags",
            "area",
            "reasoning",
            "question",
            "correct_answer_label",
            "original_prediction",
            "karl_prediction",
            "original_correct",
            "karl_correct",
            "is_correct",
            "delta_from_original",
            "outcome",
            "active_tokens_mean",
            "reconstruction_l1_mean",
        ],
    )
    write_csv(
        tables_dir / "combined_family_accuracy.csv",
        summary["family_summaries"],
        [
            "family",
            "condition",
            "rows",
            "accuracy",
            "original_accuracy",
            "accuracy_delta_from_original",
            "active_tokens_mean",
            "reconstruction_l1_mean",
            "fixed_count",
            "lost_count",
            "kept_correct_count",
            "kept_wrong_count",
        ],
    )
    write_csv(
        tables_dir / "combined_major_tag_accuracy.csv",
        summary["tag_summaries"],
        [
            "tag",
            "condition",
            "rows",
            "accuracy",
            "original_accuracy",
            "accuracy_delta_from_original",
            "active_tokens_mean",
            "reconstruction_l1_mean",
            "fixed_count",
            "lost_count",
            "kept_correct_count",
            "kept_wrong_count",
        ],
    )
    write_csv(
        tables_dir / "combined_source_accuracy.csv",
        summary["source_summaries"],
        [
            "source_subsets",
            "condition",
            "rows",
            "accuracy",
            "original_accuracy",
            "accuracy_delta_from_original",
            "active_tokens_mean",
            "reconstruction_l1_mean",
            "fixed_count",
            "lost_count",
            "kept_correct_count",
            "kept_wrong_count",
        ],
    )


def make_figures(output_dir: Path, summary: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional reporting dependency
        print(f"[combined-qwen-karl] skipped figures: {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [summary["condition_summary"][condition] for condition in CONDITION_ORDER]
    xs = [row["active_tokens_mean"] for row in rows]
    ys = [row["accuracy"] for row in rows]
    labels = list(CONDITION_ORDER)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points")
    plt.gca().invert_xaxis()
    plt.xlabel("Mean active KARL tokens")
    plt.ylabel("Qwen accuracy")
    plt.title("Combined Qwen accuracy vs KARL token budget")
    plt.tight_layout()
    plt.savefig(figures_dir / "combined_accuracy_vs_active_tokens.png", dpi=180)
    plt.close()

    tag_matrix: list[list[float]] = []
    tag_labels: list[str] = []
    by_tag_condition = {(row["tag"], row["condition"]): row for row in summary["tag_summaries"]}
    for tag in MAJOR_TAGS:
        if (tag, "original") not in by_tag_condition:
            continue
        tag_labels.append(tag)
        tag_matrix.append([by_tag_condition[(tag, condition)]["accuracy"] for condition in CONDITION_ORDER])
    if tag_matrix:
        arr = np.asarray(tag_matrix, dtype=float)
        plt.figure(figsize=(7, max(4, 0.34 * len(tag_labels))))
        plt.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        plt.colorbar(label="accuracy")
        plt.xticks(range(len(CONDITION_ORDER)), CONDITION_ORDER)
        plt.yticks(range(len(tag_labels)), tag_labels)
        plt.title("Combined tag-wise Qwen accuracy")
        plt.tight_layout()
        plt.savefig(figures_dir / "combined_tag_accuracy_heatmap.png", dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    same_base = workspace / "outputs" / "karl_mdl" / "same_video_task_control_v1"
    main_base = workspace / "outputs" / "karl_mdl" / "task_stratified_recon_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-subset", type=Path, default=workspace / "outputs" / "curated_tasks" / "subsets" / "main_balanced_300_seed17.jsonl")
    parser.add_argument("--same-video-subset", type=Path, default=workspace / "outputs" / "curated_tasks" / "subsets" / "same_video_controls_seed17.jsonl")
    parser.add_argument("--curation-summary", type=Path, default=workspace / "outputs" / "curated_tasks" / "reports" / "curated_task_data_summary.json")
    parser.add_argument("--main-original", type=Path, default=workspace / "outputs" / "baseline_main" / "predictions" / "qwen_uniform8_original.jsonl")
    parser.add_argument("--same-video-original", type=Path, default=same_base / "qwen_original_baseline" / "predictions" / "qwen_uniform8_original.jsonl")
    parser.add_argument("--main-karl-dir", type=Path, default=main_base / "qwen_karl_reconstruction_baseline" / "predictions")
    parser.add_argument("--same-video-karl-dir", type=Path, default=same_base / "qwen_karl_reconstruction_baseline" / "predictions")
    parser.add_argument("--epsilons", nargs="+", default=list(EPSILON_TAGS))
    parser.add_argument("--output-dir", type=Path, default=workspace / "outputs" / "karl_mdl" / "combined_qwen_karl_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = workspace_root()
    epsilon_tags = tuple(args.epsilons)
    main_subset = read_jsonl(resolve_path(args.main_subset, workspace))
    same_video_subset = read_jsonl(resolve_path(args.same_video_subset, workspace))
    curation_summary = read_json(resolve_path(args.curation_summary, workspace))

    rows_by_uid: dict[str, dict[str, Any]] = {}
    add_subset_rows(rows_by_uid, main_subset, "main_balanced_300_seed17")
    add_subset_rows(rows_by_uid, same_video_subset, "same_video_controls_seed17")
    combined_rows = sorted(rows_by_uid.values(), key=lambda row: row["row_uid"])

    main_original = load_prediction_map(resolve_path(args.main_original, workspace))
    same_video_original = load_prediction_map(resolve_path(args.same_video_original, workspace))
    main_karl = load_karl_prediction_maps(resolve_path(args.main_karl_dir, workspace), epsilon_tags)
    same_video_karl = load_karl_prediction_maps(resolve_path(args.same_video_karl_dir, workspace), epsilon_tags)
    original_by_source = {
        "main_balanced_300_seed17": main_original,
        "same_video_controls_seed17": same_video_original,
    }
    karl_by_source = {
        "main_balanced_300_seed17": main_karl,
        "same_video_controls_seed17": same_video_karl,
    }

    for row in combined_rows:
        uid = row["row_uid"]
        source = source_for_row(row)
        if uid not in original_by_source[source]:
            raise KeyError(f"Missing original prediction for {uid} from {source}")
        for epsilon_tag in epsilon_tags:
            if uid not in karl_by_source[source][epsilon_tag]:
                raise KeyError(f"Missing {epsilon_tag} KARL prediction for {uid} from {source}")

    construction = {
        "source_train_rows": curation_summary["source_train_rows"],
        "enriched_rows": curation_summary["enriched_rows"],
        "curated_rows": curation_summary["curated_rows"],
        "curated_unique_videos": curation_summary["curated_unique_videos"],
        "main_rows": len(main_subset),
        "main_unique_videos": len({row["video_id"] for row in main_subset}),
        "same_video_rows": len(same_video_subset),
        "same_video_unique_videos": len({row["video_id"] for row in same_video_subset}),
        "overlap_rows": len({row["row_uid"] for row in main_subset} & {row["row_uid"] for row in same_video_subset}),
        "combined_rows": len(combined_rows),
        "combined_unique_videos": len({row["video_id"] for row in combined_rows}),
        "dedupe_key": "row_uid",
        "overlap_prediction_priority": "same_video_controls_seed17",
    }

    question_rows = build_question_rows(combined_rows, original_by_source, karl_by_source, epsilon_tags)
    summary = build_summaries(question_rows, construction)
    output_dir = resolve_path(args.output_dir, workspace)
    write_summary_tables(output_dir, question_rows, summary)
    write_json(output_dir / "reports" / "combined_qwen_karl_tradeoff_summary.json", summary)
    write_text(output_dir / "reports" / "combined_qwen_karl_tradeoff_summary.md", build_markdown(summary))
    make_figures(output_dir, summary)

    print(f"[combined-qwen-karl] combined rows: {construction['combined_rows']}")
    print(f"[combined-qwen-karl] unique videos: {construction['combined_unique_videos']}")
    print(f"[combined-qwen-karl] overlap rows deduped: {construction['overlap_rows']}")
    print(f"[combined-qwen-karl] report: {output_dir / 'reports' / 'combined_qwen_karl_tradeoff_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
