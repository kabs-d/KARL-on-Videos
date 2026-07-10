#!/usr/bin/env python3
"""Run Qwen2.5-VL frame-budget calibration on curated Perception Test MCQs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


ANSWER_LETTERS = ("A", "B", "C")
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "hourvideo_workspace"
    / "data"
    / "model_cache"
    / "Qwen2.5-VL-7B-Instruct"
)
HOURVIDEO_SRC = Path(__file__).resolve().parents[2] / "hourvideo_workspace" / "src"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_hourvideo_src() -> None:
    if str(HOURVIDEO_SRC) not in sys.path:
        sys.path.insert(0, str(HOURVIDEO_SRC))


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


def build_prompt(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{letter}. {option}" for letter, option in zip(ANSWER_LETTERS, row["options"]))
    return (
        "Answer the multiple-choice video question using exactly one letter: A, B, or C.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Return only the answer letter."
    )


def parse_answer_letter(raw_output: str | None) -> str | None:
    if raw_output is None:
        return None
    text = str(raw_output).strip()
    if not text:
        return None
    patterns = (
        r"^\(?\s*([ABC])\s*\)?\.?$",
        r"\bANSWER\s*(?:IS|:|-)?\s*\(?([ABC])\)?\b",
        r"\(([ABC])\)",
        r"\b([ABC])\b",
    )
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def answer_id_to_letter(answer_id: int) -> str:
    return ANSWER_LETTERS[int(answer_id)]


def uniform_indices(frame_count: int, num_frames: int, cut_frame: int | None) -> list[int]:
    usable_count = frame_count
    if cut_frame is not None and cut_frame > 0:
        usable_count = min(usable_count, cut_frame)
    if usable_count <= 0 or num_frames <= 0:
        return []
    if num_frames >= usable_count:
        return list(range(usable_count))
    return sorted({int(round(value)) for value in np.linspace(0, usable_count - 1, num=num_frames)})


def resize_center_crop_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    width, height = pil.size
    scale = size / min(width, height)
    resized = pil.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    cropped = resized.crop((left, top, left + size, top + size))
    return np.asarray(cropped)


def sample_frames_decord(video_path: Path, num_frames: int, cut_frame: int | None, image_size: int) -> dict[str, Any]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    frame_count = len(reader)
    fps = float(reader.get_avg_fps())
    indices = uniform_indices(frame_count, num_frames, cut_frame)
    batch = reader.get_batch(indices).asnumpy() if indices else np.empty((0,))
    frames = [resize_center_crop_rgb(frame, image_size) for frame in batch]
    return {
        "backend": "decord",
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps > 0 else None,
        "sampled_frame_indices": indices,
        "sampled_timestamps_seconds": [index / fps if fps > 0 else None for index in indices],
        "sampled_frames": frames,
    }


def sample_frames_opencv(video_path: Path, num_frames: int, cut_frame: int | None, image_size: int) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = uniform_indices(frame_count, num_frames, cut_frame)
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image_bgr = capture.read()
        if not ok:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        frames.append(resize_center_crop_rgb(image_rgb, image_size))
    capture.release()
    return {
        "backend": "opencv",
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps and fps > 0 else None,
        "sampled_frame_indices": indices,
        "sampled_timestamps_seconds": [index / fps if fps and fps > 0 else None for index in indices],
        "sampled_frames": frames,
    }


def sample_frames(video_path: Path, num_frames: int, cut_frame: int | None, image_size: int) -> dict[str, Any]:
    try:
        return sample_frames_decord(video_path, num_frames, cut_frame, image_size)
    except Exception as exc:
        result = sample_frames_opencv(video_path, num_frames, cut_frame, image_size)
        result["warning"] = f"decord failed; used OpenCV fallback: {exc}"
        return result


def select_rows(rows: list[dict[str, Any]], limit: int | None, per_family_limit: int | None) -> list[dict[str, Any]]:
    if per_family_limit is not None:
        counts: Counter[str] = Counter()
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


def existing_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for row in read_jsonl(path):
        keys.add((str(row.get("row_uid")), int(row.get("num_frames", -1))))
    return keys


def load_qwen(model_name_or_path: Path | str, max_new_tokens: int):
    add_hourvideo_src()
    from hourvideo_workspace.models.qwen2_5_vl import Qwen25VLAdapter

    return Qwen25VLAdapter(
        model_name_or_path=str(model_name_or_path),
        max_new_tokens=max_new_tokens,
        local_files_only=True,
    ).load()


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record.get("is_correct") is True)
    invalid = sum(1 for record in records if record.get("parsed_prediction") is None)
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["primary_task_family"]].append(record)
    for family, family_records in sorted(grouped.items()):
        family_total = len(family_records)
        family_correct = sum(1 for record in family_records if record.get("is_correct") is True)
        family_invalid = sum(1 for record in family_records if record.get("parsed_prediction") is None)
        by_family[family] = {
            "total": family_total,
            "correct": family_correct,
            "accuracy": family_correct / family_total if family_total else 0.0,
            "invalid_output_count": family_invalid,
            "invalid_output_rate": family_invalid / family_total if family_total else 0.0,
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / total if total else 0.0,
        "by_primary_task_family": by_family,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen Perception Test Calibration",
        "",
        f"- condition: `{summary['condition']}`",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument(
        "--subset",
        type=Path,
        default=workspace_root() / "outputs" / "curated_tasks" / "subsets" / "calibration_balanced_150_seed17.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=workspace_root() / "outputs" / "baseline_calibration")
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-family-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    subset_path = args.subset if args.subset.is_absolute() else workspace / args.subset
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    condition = f"qwen_uniform{args.num_frames}_original"
    prediction_path = output_dir / "predictions" / f"{condition}.jsonl"
    summary_json_path = output_dir / "reports" / f"{condition}_summary.json"
    summary_md_path = output_dir / "reports" / f"{condition}_summary.md"

    rows = select_rows(read_jsonl(subset_path), args.limit, args.per_family_limit)
    if not rows:
        raise SystemExit("No rows selected.")

    skip_keys = existing_keys(prediction_path) if args.skip_existing else set()
    selected = [
        row
        for row in rows
        if (row["row_uid"], args.num_frames) not in skip_keys
    ]
    print(f"[calibration] subset: {subset_path}")
    print(f"[calibration] selected rows: {len(selected)} / requested {len(rows)}")
    print(f"[calibration] condition: {condition}")
    print(f"[calibration] dry_run: {args.dry_run}")

    start = time.time()
    records: list[dict[str, Any]] = []
    model = None if args.dry_run else load_qwen(args.model_name_or_path, args.max_new_tokens)

    for row in tqdm(selected, desc=condition):
        video_path = workspace / row["video_path"]
        cut_frame = row.get("cut_frame")
        cut_frame = int(cut_frame) if isinstance(cut_frame, int | float) else None
        sample_result = sample_frames(video_path, args.num_frames, cut_frame, args.image_size)
        indices = sample_result["sampled_frame_indices"]
        if cut_frame is not None and cut_frame > 0 and indices and max(indices) >= cut_frame:
            raise RuntimeError(f"cut_frame violation for {row['row_uid']}: max index {max(indices)} >= {cut_frame}")
        if len(sample_result["sampled_frames"]) != len(indices):
            raise RuntimeError(f"decoded frame count mismatch for {row['row_uid']}")

        prompt = build_prompt(row)
        raw_output = None
        parsed = None
        is_correct = None
        if not args.dry_run:
            raw_output = model.predict({"prompt": prompt, "sampled_frames": sample_result["sampled_frames"]})
            parsed = parse_answer_letter(raw_output)
            is_correct = parsed == answer_id_to_letter(int(row["answer_id"]))

        record = {
            "condition": condition,
            "num_frames": args.num_frames,
            "image_size": args.image_size,
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
            "cut_frame": row.get("cut_frame"),
            "video_backend": sample_result.get("backend"),
            "video_frame_count": sample_result.get("frame_count"),
            "video_fps": sample_result.get("fps"),
            "video_duration_seconds": sample_result.get("duration_seconds"),
            "sampled_frame_indices": indices,
            "sampled_timestamps_seconds": sample_result["sampled_timestamps_seconds"],
            "sampling_warning": sample_result.get("warning"),
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
        dry_path = output_dir / "dry_runs" / f"{condition}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        append_jsonl(dry_path, records)
        print(f"[calibration] wrote dry run rows: {dry_path}")
        return 0

    all_records = read_jsonl(prediction_path)
    condition_records = [
        record for record in all_records if record.get("condition") == condition and int(record.get("num_frames", -1)) == args.num_frames
    ]
    summary = {
        "condition": condition,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "subset": str(subset_path),
        "prediction_path": str(prediction_path),
        "model_name_or_path": str(args.model_name_or_path),
        "runtime_seconds": runtime,
        "seconds_per_example": runtime / len(records) if records else 0.0,
        "new_records_written": len(records),
        "total_condition_records": len(condition_records),
        "metrics": evaluate(condition_records),
    }
    write_json(summary_json_path, summary)
    write_text(summary_md_path, format_markdown(summary))
    print(f"[calibration] wrote predictions: {prediction_path}")
    print(f"[calibration] wrote summary: {summary_json_path}")
    print(f"[calibration] accuracy: {summary['metrics']['accuracy']:.4f}")
    print(f"[calibration] invalid_output_rate: {summary['metrics']['invalid_output_rate']:.4f}")
    print(f"[calibration] seconds_per_example: {summary['seconds_per_example']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
