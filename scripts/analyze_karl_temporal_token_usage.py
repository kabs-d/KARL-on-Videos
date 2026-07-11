#!/usr/bin/env python3
"""Summarize temporal active-token usage for KARL video-frame runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def plot_bars(path: Path, labels: list[str], values: list[float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 450
    margin_l, margin_t, margin_b, margin_r = 90, 55, 75, 35
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 20), title, fill=(0, 0, 0))

    ymin = min(values + [0.0])
    ymax = max(values + [0.0])
    pad = max(1e-8, (ymax - ymin) * 0.15)
    ymin -= pad
    ymax += pad
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    zero_y = margin_t + int((ymax - 0.0) / (ymax - ymin) * plot_h)
    zero_y = max(margin_t, min(margin_t + plot_h, zero_y))
    draw.line([margin_l, zero_y, margin_l + plot_w, zero_y], fill=(160, 160, 160))

    colors = [(76, 120, 168), (245, 133, 24), (84, 162, 75)]
    group_w = plot_w / max(1, len(labels))
    bar_w = int(group_w * 0.42)
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = margin_l + int((i + 0.5) * group_w)
        value_y = margin_t + int((ymax - value) / (ymax - ymin) * plot_h)
        value_y = max(margin_t, min(margin_t + plot_h, value_y))
        y0, y1 = sorted([zero_y, value_y])
        draw.rectangle([cx - bar_w // 2, y0, cx + bar_w // 2, y1], fill=colors[i % len(colors)])
        label_y = y0 - 20 if value >= 0 else y1 + 4
        draw.text((cx - bar_w // 2, label_y), f"{value:.3f}", fill=(0, 0, 0))
        draw.text((cx - 28, margin_t + plot_h + 14), label, fill=(0, 0, 0))

    draw.line([margin_l, margin_t, margin_l, margin_t + plot_h], fill=(0, 0, 0))
    draw.line([margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h], fill=(0, 0, 0))
    image.save(path)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_epsilon: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_epsilon[row["epsilon_tag"]].append(row)

    out: list[dict[str, Any]] = []
    for epsilon_tag, eps_rows in sorted(by_epsilon.items()):
        corr_values = [safe_float(row["corr_active_frame_diff"]) for row in eps_rows]
        active_ranges = [safe_float(row["active_range"]) for row in eps_rows]
        epsilon = f"{int(epsilon_tag.replace('eps_', '')) / 100:.2f}"
        out.append(
            {
                "epsilon": epsilon,
                "videos": len(eps_rows),
                "frame_epsilon_rows": len(eps_rows) * 8,
                "mean_active_token_std": f"{mean_finite([safe_float(row['active_std']) for row in eps_rows]):.4f}",
                "mean_active_token_range": f"{mean_finite(active_ranges):.4f}",
                "zero_range_videos": sum(1 for value in active_ranges if value == 0.0),
                "corr_ge_0_5_videos": sum(1 for value in corr_values if math.isfinite(value) and value >= 0.5),
                "corr_le_minus_0_5_videos": sum(1 for value in corr_values if math.isfinite(value) and value <= -0.5),
                "mean_frame_diff": f"{mean_finite([safe_float(row['frame_diff_mean']) for row in eps_rows]):.4f}",
                "mean_corr_active_tokens_frame_diff": f"{mean_finite(corr_values):.4f}",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.input_csv)
    summary = summarize(rows)
    fields = [
        "epsilon",
        "videos",
        "frame_epsilon_rows",
        "mean_active_token_std",
        "mean_active_token_range",
        "zero_range_videos",
        "corr_ge_0_5_videos",
        "corr_le_minus_0_5_videos",
        "mean_frame_diff",
        "mean_corr_active_tokens_frame_diff",
    ]
    write_csv(args.output_dir / "tables" / "temporal_token_usage_summary.csv", summary, fields)
    labels = [f"eps_{int(round(float(row['epsilon']) * 100)):03d}" for row in summary]
    plot_bars(
        args.output_dir / "figures" / "active_variation_by_epsilon.png",
        labels,
        [float(row["mean_active_token_std"]) for row in summary],
        "Within-video active-token variation",
    )
    plot_bars(
        args.output_dir / "figures" / "corr_active_frame_diff_by_epsilon.png",
        labels,
        [float(row["mean_corr_active_tokens_frame_diff"]) for row in summary],
        "Per-video corr(active tokens, frame difference)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
