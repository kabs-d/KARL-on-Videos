#!/usr/bin/env python3
"""Analyze between-token diversity/collapse in KARL latent attention maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


GRID = 16
GRID_CELLS = GRID * GRID


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def normalize_rows(maps: np.ndarray) -> np.ndarray:
    flat = np.asarray(maps, dtype=np.float64).reshape(maps.shape[0], -1)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat[flat < 0] = 0
    totals = flat.sum(axis=1, keepdims=True)
    return np.divide(flat, totals, out=np.zeros_like(flat), where=totals > 0)


def standardized_rows(flat: np.ndarray) -> np.ndarray:
    centered = flat - flat.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(axis=1, keepdims=True))
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)


def center_of_mass(flat: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    y = flat @ yy.reshape(-1)
    x = flat @ xx.reshape(-1)
    return np.stack([y, x], axis=1)


def peak_positions(flat: np.ndarray) -> np.ndarray:
    indices = np.argmax(flat, axis=1)
    return np.stack([indices // GRID, indices % GRID], axis=1).astype(np.float64)


def top_masks(flat: np.ndarray, top_k: int) -> np.ndarray:
    k = min(max(1, top_k), flat.shape[1])
    indices = np.argpartition(flat, -k, axis=1)[:, -k:]
    masks = np.zeros(flat.shape, dtype=bool)
    row_indices = np.arange(flat.shape[0])[:, None]
    masks[row_indices, indices] = True
    return masks


def summarize_values(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def sampled_pair_indices(n: int, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(n, k=1)
    if left.shape[0] <= max_pairs:
        return left, right
    selected = rng.choice(left.shape[0], size=max_pairs, replace=False)
    return left[selected], right[selected]


def pair_metrics(flat: np.ndarray, left: np.ndarray, right: np.ndarray, top_k: int) -> dict[str, np.ndarray]:
    standardized = standardized_rows(flat)
    corr = np.sum(standardized[left] * standardized[right], axis=1)
    masks = top_masks(flat, top_k)
    intersection = np.logical_and(masks[left], masks[right]).sum(axis=1).astype(np.float64)
    union = np.logical_or(masks[left], masks[right]).sum(axis=1).astype(np.float64)
    top_iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    centers = center_of_mass(flat)
    center_distance = np.linalg.norm(centers[left] - centers[right], axis=1)
    peaks = peak_positions(flat)
    peak_distance = np.linalg.norm(peaks[left] - peaks[right], axis=1)
    return {
        "corr": corr,
        "top_iou": top_iou,
        "center_distance": center_distance,
        "peak_distance": peak_distance,
    }


def plot_bars(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 480
    margin_l, margin_t, margin_b, margin_r = 90, 55, 90, 35
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 18), title, fill=(0, 0, 0))
    max_value = max(values + [1e-8])
    max_value *= 1.15
    for i in range(6):
        y = margin_t + int(plot_h * i / 5)
        draw.line([margin_l, y, margin_l + plot_w, y], fill=(225, 225, 225))
    group_w = plot_w / max(1, len(labels))
    bar_w = int(group_w * 0.34)
    for i, label in enumerate(labels):
        cx = margin_l + int((i + 0.5) * group_w)
        value = values[i]
        bar_h = int(plot_h * value / max_value)
        x0 = cx - bar_w // 2
        x1 = cx + bar_w // 2
        y0 = margin_t + plot_h - bar_h
        y1 = margin_t + plot_h
        draw.rectangle([x0, y0, x1, y1], fill=(76, 120, 168))
        draw.text((x0, max(margin_t, y0 - 18)), f"{value:.3f}", fill=(0, 0, 0))
        draw.text((cx - 28, margin_t + plot_h + 14), label, fill=(0, 0, 0))
    draw.text((margin_l, height - 32), ylabel, fill=(0, 0, 0))
    draw.line([margin_l, margin_t, margin_l, margin_t + plot_h], fill=(0, 0, 0))
    draw.line([margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h], fill=(0, 0, 0))
    image.save(path)


def plot_scatter(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 520
    margin_l, margin_t, margin_b, margin_r = 85, 55, 75, 35
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 18), title, fill=(0, 0, 0))
    xs = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    eps = [str(row["epsilon_tag"]) for row in rows]
    colors = {"eps_003": (76, 120, 168), "eps_005": (245, 133, 24), "eps_007": (84, 162, 75)}
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    for i in range(6):
        y = margin_t + int(plot_h * i / 5)
        draw.line([margin_l, y, margin_l + plot_w, y], fill=(230, 230, 230))
    for x, y, tag in zip(xs, ys, eps):
        px = margin_l + int((x - xmin) / (xmax - xmin) * plot_w)
        py = margin_t + plot_h - int((y - ymin) / (ymax - ymin) * plot_h)
        color = colors.get(tag, (80, 80, 80))
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    draw.line([margin_l, margin_t, margin_l, margin_t + plot_h], fill=(0, 0, 0))
    draw.line([margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h], fill=(0, 0, 0))
    draw.text((margin_l, height - 34), x_key, fill=(0, 0, 0))
    draw.text((12, margin_t + 8), y_key, fill=(0, 0, 0))
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument("--max-pairs-per-frame", type=int, default=5000)
    parser.add_argument("--top-k-cells", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    input_dir = resolve_path(args.input_dir, workspace)
    output_dir = resolve_path(args.output_dir, workspace)
    frame_path = input_dir / "frame_metrics.jsonl"
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)

    rng = np.random.default_rng(args.seed)
    frame_rows_in = [row for row in read_jsonl(frame_path) if row.get("attention_map_path")]
    if args.limit_frames is not None:
        frame_rows_in = frame_rows_in[: args.limit_frames]
    frame_summaries: list[dict[str, Any]] = []

    for row in frame_rows_in:
        map_path = resolve_path(Path(row["attention_map_path"]), workspace)
        z = np.load(map_path)
        attention_maps = np.asarray(z["encoder_latent_to_input_grid_attn_16x16"], dtype=np.float64)
        active_indices = np.asarray(z["active_token_indices"], dtype=np.int64)
        if attention_maps.ndim != 3 or attention_maps.shape[-2:] != (GRID, GRID):
            raise RuntimeError(f"Unexpected attention map shape in {map_path}: {attention_maps.shape}")

        n = int(attention_maps.shape[0])
        possible_pairs = n * (n - 1) // 2
        summary: dict[str, Any] = {
            "epsilon_tag": row["epsilon_tag"],
            "epsilon": row["epsilon"],
            "video_id": row["video_id"],
            "video_sample_uid": row.get("video_sample_uid", ""),
            "frame_position": int(row["frame_position"]),
            "frame_index": int(row["frame_index"]),
            "active_token_count": n,
            "possible_pairs": possible_pairs,
            "sampled_pairs": 0,
            "attention_map_path": row["attention_map_path"],
        }
        if n < 2:
            frame_summaries.append(summary)
            continue

        left, right = sampled_pair_indices(n, args.max_pairs_per_frame, rng)
        summary["sampled_pairs"] = int(left.shape[0])
        attention_flat = normalize_rows(attention_maps)
        attention_pair = pair_metrics(attention_flat, left, right, args.top_k_cells)
        for metric in ["corr", "top_iou", "center_distance", "peak_distance"]:
            summary.update(summarize_values(np.asarray(attention_pair[metric], dtype=np.float64), f"attention_pair_{metric}"))
        summary.update(
            {
                "attention_pair_distinctness_mean": float(1.0 - np.mean(attention_pair["corr"])),
                "active_token_index_min": int(active_indices.min()) if active_indices.size else "",
                "active_token_index_max": int(active_indices.max()) if active_indices.size else "",
            }
        )
        frame_summaries.append(summary)

    frame_fields = [
        "epsilon_tag", "epsilon", "video_id", "video_sample_uid", "frame_position", "frame_index",
        "active_token_count", "possible_pairs", "sampled_pairs",
        "attention_pair_corr_mean", "attention_pair_corr_median", "attention_pair_corr_q25", "attention_pair_corr_q75",
        "attention_pair_top_iou_mean", "attention_pair_top_iou_median", "attention_pair_top_iou_q25", "attention_pair_top_iou_q75",
        "attention_pair_center_distance_mean", "attention_pair_center_distance_median", "attention_pair_center_distance_q25", "attention_pair_center_distance_q75",
        "attention_pair_peak_distance_mean", "attention_pair_peak_distance_median", "attention_pair_peak_distance_q25", "attention_pair_peak_distance_q75",
        "attention_pair_distinctness_mean",
        "active_token_index_min", "active_token_index_max", "attention_map_path",
    ]
    write_csv(output_dir / "tables" / "latent_frame_diversity_summary.csv", frame_summaries, frame_fields)

    by_epsilon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_summaries:
        by_epsilon[str(row["epsilon_tag"])].append(row)

    epsilon_rows: list[dict[str, Any]] = []
    metric_keys = [
        "active_token_count",
        "sampled_pairs",
        "attention_pair_corr_mean",
        "attention_pair_top_iou_mean",
        "attention_pair_center_distance_mean",
        "attention_pair_distinctness_mean",
    ]
    for epsilon_tag, rows in sorted(by_epsilon.items()):
        item: dict[str, Any] = {"epsilon_tag": epsilon_tag, "frame_rows": len(rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in rows if row.get(key) not in ("", None) and not math.isnan(float(row[key]))])
            item[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
            item[f"{key}_median"] = float(np.median(values)) if values.size else float("nan")
        epsilon_rows.append(item)
    epsilon_fields = ["epsilon_tag", "frame_rows"]
    for key in metric_keys:
        epsilon_fields.extend([f"{key}_mean", f"{key}_median"])
    write_csv(output_dir / "tables" / "latent_epsilon_diversity_summary.csv", epsilon_rows, epsilon_fields)

    summary = {
        "frame_rows": len(frame_summaries),
        "max_pairs_per_frame": args.max_pairs_per_frame,
        "top_k_cells": args.top_k_cells,
        "seed": args.seed,
        "by_epsilon": {row["epsilon_tag"]: row for row in epsilon_rows},
    }
    write_json(output_dir / "reports" / "latent_diversity_summary.json", summary)

    labels = [row["epsilon_tag"] for row in epsilon_rows]
    plot_bars(
        output_dir / "figures" / "pairwise_similarity_by_epsilon.png",
        labels,
        [float(row["attention_pair_corr_mean_mean"]) for row in epsilon_rows],
        "Mean pairwise attention-map correlation",
        "higher = more similar / more collapsed",
    )
    plot_bars(
        output_dir / "figures" / "pairwise_overlap_by_epsilon.png",
        labels,
        [float(row["attention_pair_top_iou_mean_mean"]) for row in epsilon_rows],
        "Mean pairwise top-cell IoU",
        "higher = more spatial overlap",
    )
    plot_bars(
        output_dir / "figures" / "pairwise_center_distance_by_epsilon.png",
        labels,
        [float(row["attention_pair_center_distance_mean_mean"]) for row in epsilon_rows],
        "Mean pairwise center distance",
        "higher = more spatially spread out",
    )
    plot_scatter(
        output_dir / "figures" / "diversity_vs_active_count.png",
        frame_summaries,
        "active_token_count",
        "attention_pair_corr_mean",
        "Active token count vs attention-map pairwise similarity",
    )

    lines = [
        "# KARL Latent Diversity / Collapse Analysis",
        "",
        f"- frame-epsilon rows: {len(frame_summaries)}",
        f"- sampled pairs per frame cap: {args.max_pairs_per_frame}",
        f"- top-k cells for overlap: {args.top_k_cells}",
        "",
        "| epsilon | active tokens | attention corr | attention IoU | attention dist | attention distinct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in epsilon_rows:
        lines.append(
            f"| {row['epsilon_tag']} | {row['active_token_count_mean']:.2f} | "
            f"{row['attention_pair_corr_mean_mean']:.4f} | "
            f"{row['attention_pair_top_iou_mean_mean']:.4f} | "
            f"{row['attention_pair_center_distance_mean_mean']:.3f} | "
            f"{row['attention_pair_distinctness_mean_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Higher pairwise correlation / top-IoU means active tokens are more similar to each other.",
            "- Higher center distance means active tokens cover more spatially separated regions.",
            "- Higher distinctness is `1 - mean pairwise correlation`.",
        ]
    )
    write_text(output_dir / "reports" / "latent_diversity_summary.md", "\n".join(lines) + "\n")
    print(f"[latent-diversity] wrote frame summary: {output_dir / 'tables' / 'latent_frame_diversity_summary.csv'}")
    print(f"[latent-diversity] wrote epsilon summary: {output_dir / 'tables' / 'latent_epsilon_diversity_summary.csv'}")
    print(f"[latent-diversity] wrote report: {output_dir / 'reports' / 'latent_diversity_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
