#!/usr/bin/env python3
"""Run Qwen2.5-VL on KARL reconstructed Perception Test frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen_perception_calibration import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    answer_id_to_letter,
    build_prompt,
    evaluate,
    load_qwen,
    parse_answer_letter,
)

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


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def epsilon_tag(epsilon: float) -> str:
    return f"eps_{int(round(epsilon * 100)):03d}"


def select_rows(rows: list[dict[str, Any]], limit: int | None, per_family_limit: int | None) -> list[dict[str, Any]]:
    if per_family_limit is not None:
        counts: dict[str, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        for row in rows:
            family = row["primary_task_family"]
            if counts[family] >= per_family_limit:
                continue
            selected.append(row)
            counts[family] += 1
        return selected
    if limit is not None:
        return rows[:limit]
    return rows


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("row_uid")) for row in read_jsonl(path)}


def load_link_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    return {(str(row["row_uid"]), str(row["epsilon_tag"])): row for row in rows}


def load_question_metric_rows(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {(str(row["row_uid"]), str(row["epsilon_tag"])): row for row in rows}


def load_video_metrics(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    return {(str(row["video_sample_uid"]), str(row["epsilon_tag"])): row for row in rows}


def load_reconstruction_frames(paths: list[str], workspace: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for rel_path in paths:
        image_path = resolve_path(Path(rel_path), workspace)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing KARL reconstruction frame: {image_path}")
        image = Image.open(image_path).convert("RGB")
        frames.append(np.asarray(image))
    return frames


def summarize_preservation(karl_rows: list[dict[str, Any]], original_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_by_uid = {str(row["row_uid"]): row for row in original_rows}
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in karl_rows:
        by_tag[str(row["epsilon_tag"])].append(row)

    def mean_bool(values: list[bool]) -> float | None:
        return float(sum(1 for value in values if value) / len(values)) if values else None

    summary: dict[str, Any] = {
        "original_rows": len(original_rows),
        "karl_rows": len(karl_rows),
        "by_epsilon": {},
    }
    for tag, rows in sorted(by_tag.items()):
        comparable = [(row, original_by_uid.get(str(row["row_uid"]))) for row in rows]
        comparable = [(row, original) for row, original in comparable if original is not None]
        same_prediction = [
            row.get("parsed_prediction") is not None
            and original.get("parsed_prediction") is not None
            and row.get("parsed_prediction") == original.get("parsed_prediction")
            for row, original in comparable
        ]
        original_correct = [(row, original) for row, original in comparable if original.get("is_correct") is True]
        original_wrong = [(row, original) for row, original in comparable if original.get("is_correct") is False]
        summary["by_epsilon"][tag] = {
            "rows": len(rows),
            "comparable_original_rows": len(comparable),
            "same_prediction_rate": mean_bool(same_prediction),
            "original_correct_kept_correct_rate": mean_bool([row.get("is_correct") is True for row, _ in original_correct]),
            "original_correct_to_wrong_rate": mean_bool([row.get("is_correct") is False for row, _ in original_correct]),
            "original_wrong_to_correct_rate": mean_bool([row.get("is_correct") is True for row, _ in original_wrong]),
        }
    return summary


def tags_of(row: dict[str, Any]) -> set[str]:
    tags = row.get("tag", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def summarize_tradeoff(
    karl_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    fresh_exclude_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    original_by_uid = {str(row["row_uid"]): row for row in original_rows}
    excluded_uids = {str(row["row_uid"]) for row in fresh_exclude_rows}
    all_uids = set(original_by_uid)
    fresh_uids = all_uids - excluded_uids
    by_epsilon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in karl_rows:
        by_epsilon[str(row["epsilon_tag"])].append(row)

    def acc(rows: list[dict[str, Any]]) -> float | None:
        return rate(sum(1 for row in rows if row.get("is_correct") is True), len(rows))

    def summarize_rows(rows: list[dict[str, Any]], uids: set[str] | None = None) -> dict[str, Any]:
        selected = [row for row in rows if uids is None or str(row["row_uid"]) in uids]
        return {
            "rows": len(selected),
            "accuracy": acc(selected),
            "active_tokens_mean": mean([row.get("active_tokens_mean") for row in selected]),
            "reconstruction_l1_mean": mean([row.get("reconstruction_l1_mean") for row in selected]),
        }

    out: dict[str, Any] = {
        "original": {
            "rows": len(original_rows),
            "fresh_rows": len([row for row in original_rows if str(row["row_uid"]) in fresh_uids]),
            "accuracy": acc(original_rows),
            "fresh_accuracy": acc([row for row in original_rows if str(row["row_uid"]) in fresh_uids]),
        },
        "fresh_exclude_rows": len(excluded_uids & all_uids),
        "fresh_rows": len(fresh_uids),
        "by_epsilon": {},
        "by_epsilon_family": {},
        "by_epsilon_major_tag": {},
    }

    for tag, rows in sorted(by_epsilon.items()):
        out["by_epsilon"][tag] = {
            **summarize_rows(rows),
            "fresh": summarize_rows(rows, fresh_uids),
        }

        family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            family_groups[str(row.get("primary_task_family", ""))].append(row)
        for family, family_rows in sorted(family_groups.items()):
            out["by_epsilon_family"][f"{tag}:{family}"] = {
                "epsilon_tag": tag,
                "family": family,
                **summarize_rows(family_rows),
                "fresh": summarize_rows(family_rows, fresh_uids),
            }

        for major_tag in MAJOR_TAGS:
            tag_rows = [row for row in rows if major_tag in tags_of(row)]
            if not tag_rows:
                continue
            out["by_epsilon_major_tag"][f"{tag}:{major_tag}"] = {
                "epsilon_tag": tag,
                "tag": major_tag,
                **summarize_rows(tag_rows),
                "fresh": summarize_rows(tag_rows, fresh_uids),
            }
    return out


def format_tradeoff_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Main-300 Qwen-KARL Tradeoff",
        "",
        f"- original rows: {summary['original']['rows']}",
        f"- fresh-only rows: {summary['fresh_rows']}",
        f"- exact overlaps excluded from fresh-only view: {summary['fresh_exclude_rows']}",
        f"- original accuracy: {fmt(summary['original']['accuracy'])}",
        f"- original fresh-only accuracy: {fmt(summary['original']['fresh_accuracy'])}",
        "",
        "## Global",
        "",
        "| epsilon | rows | acc | fresh rows | fresh acc | active mean | L1 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, row in sorted(summary["by_epsilon"].items()):
        fresh = row["fresh"]
        lines.append(
            f"| {tag} | {row['rows']} | {fmt(row['accuracy'])} | "
            f"{fresh['rows']} | {fmt(fresh['accuracy'])} | "
            f"{fmt(row['active_tokens_mean'], 2)} | {fmt(row['reconstruction_l1_mean'], 5)} |"
        )

    lines.extend(["", "## By Family", "", "| epsilon | family | rows | acc | fresh rows | fresh acc | active mean | L1 mean |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for _key, row in sorted(summary["by_epsilon_family"].items()):
        fresh = row["fresh"]
        lines.append(
            f"| {row['epsilon_tag']} | {row['family']} | {row['rows']} | {fmt(row['accuracy'])} | "
            f"{fresh['rows']} | {fmt(fresh['accuracy'])} | "
            f"{fmt(row['active_tokens_mean'], 2)} | {fmt(row['reconstruction_l1_mean'], 5)} |"
        )

    lines.extend(["", "## By Major Tag", "", "| epsilon | tag | rows | acc | fresh rows | fresh acc | active mean | L1 mean |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for _key, row in sorted(summary["by_epsilon_major_tag"].items()):
        fresh = row["fresh"]
        lines.append(
            f"| {row['epsilon_tag']} | {row['tag']} | {row['rows']} | {fmt(row['accuracy'])} | "
            f"{fresh['rows']} | {fmt(fresh['accuracy'])} | "
            f"{fmt(row['active_tokens_mean'], 2)} | {fmt(row['reconstruction_l1_mean'], 5)} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_preservation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen KARL Reconstruction vs Original",
        "",
        f"- original rows: {summary['original_rows']}",
        f"- KARL rows: {summary['karl_rows']}",
        "",
        "| epsilon | rows | comparable | same prediction | orig correct kept | orig correct lost | orig wrong fixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, metrics in summary["by_epsilon"].items():
        lines.append(
            f"| {tag} | {metrics['rows']} | {metrics['comparable_original_rows']} | "
            f"{metrics['same_prediction_rate'] if metrics['same_prediction_rate'] is not None else float('nan'):.4f} | "
            f"{metrics['original_correct_kept_correct_rate'] if metrics['original_correct_kept_correct_rate'] is not None else float('nan'):.4f} | "
            f"{metrics['original_correct_to_wrong_rate'] if metrics['original_correct_to_wrong_rate'] is not None else float('nan'):.4f} | "
            f"{metrics['original_wrong_to_correct_rate'] if metrics['original_wrong_to_correct_rate'] is not None else float('nan'):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen on KARL Reconstructions",
        "",
        f"- condition: `{summary['condition']}`",
        f"- epsilon: {summary['epsilon']}",
        f"- epsilon tag: `{summary['epsilon_tag']}`",
        f"- num frames: {summary['num_frames']}",
        f"- image size: {summary['image_size']}",
        f"- total evaluated: {summary['metrics']['total']}",
        f"- accuracy: {summary['metrics']['accuracy']:.4f}",
        f"- invalid output rate: {summary['metrics']['invalid_output_rate']:.4f}",
        f"- runtime seconds: {summary['runtime_seconds']:.2f}",
        f"- seconds per example: {summary['seconds_per_example']:.2f}",
        "",
        "## By Family",
        "",
        "| family | n | accuracy | invalid |",
        "|---|---:|---:|---:|",
    ]
    for family, metrics in summary["metrics"]["by_primary_task_family"].items():
        lines.append(
            f"| {family} | {metrics['total']} | {metrics['accuracy']:.4f} | "
            f"{metrics['invalid_output_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    default_run_dir = workspace / "outputs" / "karl_mdl" / "same_video_task_control_v1"
    output_dir = default_run_dir / "qwen_karl_reconstruction_baseline"
    default_karl_dir = default_run_dir / "karl_recon_attn_v2_video_cache"
    default_original = default_run_dir / "qwen_original_baseline" / "predictions" / "qwen_uniform8_original.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--subset",
        type=Path,
        default=workspace / "outputs" / "curated_tasks" / "subsets" / "same_video_controls_seed17.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--question-video-links", type=Path, default=default_karl_dir / "question_video_links.jsonl")
    parser.add_argument("--question-metrics", type=Path, default=None)
    parser.add_argument("--native-video-metrics", type=Path, default=default_karl_dir / "video_metrics.jsonl")
    parser.add_argument("--original-baseline-predictions", type=Path, default=default_original)
    parser.add_argument(
        "--fresh-exclude-subset",
        type=Path,
        default=workspace / "outputs" / "curated_tasks" / "subsets" / "same_video_controls_seed17.jsonl",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.03, 0.05, 0.07])
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-family-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    subset_path = resolve_path(args.subset, workspace)
    output_dir = resolve_path(args.output_dir, workspace)
    links_path = resolve_path(args.question_video_links, workspace)
    question_metrics_path = resolve_path(args.question_metrics, workspace) if args.question_metrics is not None else None
    native_video_metrics_path = resolve_path(args.native_video_metrics, workspace)
    original_baseline_path = resolve_path(args.original_baseline_predictions, workspace)
    fresh_exclude_path = resolve_path(args.fresh_exclude_subset, workspace) if args.fresh_exclude_subset is not None else None
    qwen_dir = output_dir / "predictions"
    report_dir = output_dir / "reports"

    rows = select_rows(read_jsonl(subset_path), args.limit, args.per_family_limit)
    question_metrics_by_key = load_question_metric_rows(question_metrics_path)
    links_by_key = {} if question_metrics_by_key else load_link_rows(links_path)
    native_video_by_key = load_video_metrics(native_video_metrics_path)
    model = None if args.dry_run else load_qwen(args.model_name_or_path, args.max_new_tokens)

    for epsilon in args.epsilons:
        tag = epsilon_tag(epsilon)
        condition = f"qwen_uniform{args.num_frames}_karl_{tag}"
        prediction_path = qwen_dir / f"{condition}.jsonl"
        summary_json_path = report_dir / f"{condition}_summary.json"
        summary_md_path = report_dir / f"{condition}_summary.md"
        skip_keys = existing_keys(prediction_path) if args.skip_existing else set()
        selected = [row for row in rows if row["row_uid"] not in skip_keys]
        records: list[dict[str, Any]] = []
        start = time.time()

        for row in tqdm(selected, desc=condition):
            metric_row = question_metrics_by_key.get((row["row_uid"], tag))
            if metric_row is not None:
                link_row = metric_row
                native_video_row = metric_row
            else:
                link_row = links_by_key.get((row["row_uid"], tag))
                if link_row is None:
                    raise KeyError(f"Missing KARL reconstruction row for {row['row_uid']} {tag}")
                native_video_row = native_video_by_key.get((str(link_row["video_sample_uid"]), tag), {})
            frames = load_reconstruction_frames(link_row.get("reconstruction_paths", []), workspace)
            if len(frames) != args.num_frames:
                raise RuntimeError(f"Expected {args.num_frames} reconstructed frames for {row['row_uid']} {tag}; got {len(frames)}")

            prompt = build_prompt(row)
            raw_output = None
            parsed = None
            is_correct = None
            if not args.dry_run:
                raw_output = model.predict({"prompt": prompt, "sampled_frames": frames})
                parsed = parse_answer_letter(raw_output)
                is_correct = parsed == answer_id_to_letter(int(row["answer_id"]))

            record = {
                "condition": condition,
                "epsilon": epsilon,
                "epsilon_tag": tag,
                "num_frames": args.num_frames,
                "image_size": args.image_size,
                "video_sample_uid": link_row.get("video_sample_uid"),
                "row_uid": row["row_uid"],
                "split": row.get("split"),
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "video_path": row["video_path"],
                "primary_task_family": row["primary_task_family"],
                "family_memberships": row.get("family_memberships", []),
                "area": row.get("area"),
                "reasoning": row.get("reasoning"),
                "tag": row.get("tag", []),
                "question": row["question"],
                "options": row["options"],
                "answer_id": row["answer_id"],
                "correct_answer_label": answer_id_to_letter(int(row["answer_id"])),
                "active_tokens_mean": native_video_row.get("active_tokens_mean"),
                "active_tokens_max": native_video_row.get("active_tokens_max"),
                "active_tokens_min": native_video_row.get("active_tokens_min"),
                "reconstruction_l1_mean": native_video_row.get("reconstruction_l1_mean"),
                "reconstruction_l1_max": native_video_row.get("reconstruction_l1_max"),
                "reconstruction_paths": link_row.get("reconstruction_paths", []),
                "attention_map_paths": link_row.get("attention_map_paths", []),
                "attention_overlay_paths": link_row.get("attention_overlay_paths", []),
                "sampled_frame_indices": link_row.get("sampled_frame_indices", []),
                "sampled_timestamps_seconds": link_row.get("sampled_timestamps_seconds", []),
                "prompt": prompt,
                "raw_output": raw_output,
                "parsed_prediction": parsed,
                "is_correct": is_correct,
            }
            records.append(record)
            if not args.dry_run:
                append_jsonl(prediction_path, [record])

        runtime = time.time() - start
        if args.dry_run:
            dry_path = qwen_dir / "dry_runs" / f"{condition}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
            append_jsonl(dry_path, records)
            print(f"[qwen-karl] wrote dry run rows: {dry_path}")
            continue

        all_records = read_jsonl(prediction_path)
        metrics = evaluate(all_records)
        summary = {
            "condition": condition,
            "epsilon": epsilon,
            "epsilon_tag": tag,
            "num_frames": args.num_frames,
            "image_size": args.image_size,
            "question_video_links": str(links_path) if not question_metrics_by_key else None,
            "question_metrics": str(question_metrics_path) if question_metrics_path is not None else None,
            "native_video_metrics": str(native_video_metrics_path),
            "prediction_path": str(prediction_path),
            "runtime_seconds": runtime,
            "seconds_per_example": runtime / len(records) if records else 0.0,
            "metrics": metrics,
        }
        write_json(summary_json_path, summary)
        write_text(summary_md_path, format_markdown(summary))
        print(f"[qwen-karl] wrote predictions: {prediction_path}")
        print(f"[qwen-karl] wrote summary: {summary_md_path}")

    if not args.dry_run and original_baseline_path.exists():
        all_karl_records: list[dict[str, Any]] = []
        for path in sorted(qwen_dir.glob(f"qwen_uniform{args.num_frames}_karl_eps_*.jsonl")):
            all_karl_records.extend(read_jsonl(path))
        original_rows = read_jsonl(original_baseline_path)
        preservation = summarize_preservation(all_karl_records, original_rows)
        write_json(report_dir / "qwen_karl_vs_original_summary.json", preservation)
        write_text(report_dir / "qwen_karl_vs_original_summary.md", format_preservation_markdown(preservation))
        print(f"[qwen-karl] wrote comparison summary: {report_dir / 'qwen_karl_vs_original_summary.md'}")
        fresh_exclude_rows = read_jsonl(fresh_exclude_path) if fresh_exclude_path is not None and fresh_exclude_path.exists() else []
        tradeoff = summarize_tradeoff(all_karl_records, original_rows, fresh_exclude_rows)
        write_json(report_dir / "main300_qwen_karl_tradeoff_summary.json", tradeoff)
        write_text(report_dir / "main300_qwen_karl_tradeoff_summary.md", format_tradeoff_markdown(tradeoff))
        print(f"[qwen-karl] wrote tradeoff summary: {report_dir / 'main300_qwen_karl_tradeoff_summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
