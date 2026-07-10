"""Render clean Direction 1 KARL read-attention assets.

This script packages selected token-wise read-attention maps for the
manual object-like/persistent-attention analysis. It intentionally renders
attention-only heatmaps without text overlays.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EPSILON = "eps_007"
IMAGE_SIZE = 256
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

FIRST_FRAME_TOKENS: dict[str, list[tuple[int, str]]] = {
    "video_76": [
        (36, "compact attention on a cup on the table"),
        (38, "compact attention on a cup on the table"),
        (39, "compact attention on a cup on the table"),
        (42, "compact attention on a cup on the table"),
        (159, "compact attention on a cup on the table"),
        (132, "attention concentrated near the hand"),
        (158, "attention spread over the three-cup group"),
    ],
    "video_1614": [
        (0, "cup"),
        (3, "table leg"),
        (4, "cup"),
        (17, "cup"),
        (20, "cup"),
        (25, "table edge"),
        (35, "cup"),
        (53, "cup"),
        (103, "cup"),
    ],
}

TEMPORAL_TOKENS: dict[int, str] = {
    36: "concentrated near the left-cup spatial region",
    39: "concentrated near a cup-like spatial region",
    42: "concentrated near a cup-like spatial region",
    43: "concentrated near the shirt/t-shirt spatial region",
    159: "concentrated near a cup-like spatial region when active",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent / "ShuBERT" / "perception_test_workspace"


def attention_base() -> Path:
    return (
        workspace_root()
        / "outputs"
        / "karl_mdl"
        / "same_video_task_control_v1"
        / "karl_read_write_analysis_v1"
        / "attention_maps"
        / EPSILON
    )


def frame_base() -> Path:
    return (
        workspace_root()
        / "outputs"
        / "karl_mdl"
        / "same_video_task_control_v1"
        / "karl_read_write_analysis_v1"
        / "_analysis_cache"
        / "original_frames"
        / EPSILON
    )


def output_base() -> Path:
    return repo_root() / "results" / "direction1_object_read_attention_v1"


def attention_to_rgb(attention: np.ndarray) -> Image.Image:
    """Convert a 16x16 attention map to a contrast-normalized RGB heatmap."""
    attn = np.asarray(attention, dtype=np.float32)
    finite = attn[np.isfinite(attn)]
    if finite.size == 0:
        norm = np.zeros_like(attn, dtype=np.float32)
    else:
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.5))
        if hi <= lo:
            lo = float(finite.min())
            hi = float(finite.max())
        if hi <= lo:
            norm = np.zeros_like(attn, dtype=np.float32)
        else:
            norm = np.clip((attn - lo) / (hi - lo), 0.0, 1.0)

    norm = np.power(norm, 0.65)
    stops = np.asarray(
        [
            [0.00, 0, 0, 0],
            [0.12, 38, 0, 75],
            [0.28, 0, 45, 170],
            [0.48, 0, 190, 220],
            [0.68, 255, 220, 0],
            [0.86, 255, 75, 0],
            [1.00, 255, 255, 255],
        ],
        dtype=np.float32,
    )
    flat = norm.reshape(-1)
    channels = [np.interp(flat, stops[:, 0], stops[:, c]).reshape(norm.shape) for c in (1, 2, 3)]
    rgb = np.stack(channels, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE_BICUBIC)


def load_attention(npz_path: Path, latent_index: int) -> tuple[np.ndarray | None, int | None, float | None]:
    with np.load(npz_path) as data:
        active = data["active_token_indices"].astype(int).tolist()
        if latent_index not in active:
            return None, None, None
        active_rank = active.index(latent_index)
        read_maps = data["encoder_latent_to_input_grid_attn_16x16"]
        halt = float(data["halt_probabilities"][latent_index])
        return read_maps[active_rank], active_rank, halt


def copy_original_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE_LANCZOS).save(dst)


def render_first_frame_assets(rows: list[dict[str, Any]]) -> None:
    for video_id, token_rows in FIRST_FRAME_TOKENS.items():
        npz_path = attention_base() / video_id / "frame_000_idx_000000.npz"
        original_path = frame_base() / video_id / "frame_000_idx_000000.png"
        original_out = output_base() / "original_frames" / f"{video_id}_frame_000.png"
        copy_original_frame(original_path, original_out)

        for latent_index, description in token_rows:
            attention, active_rank, halt = load_attention(npz_path, latent_index)
            if attention is None:
                out_rel = ""
                active = False
            else:
                out = output_base() / "attention_heatmaps" / "first_frame" / video_id / f"latent_{latent_index:03d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                attention_to_rgb(attention).save(out)
                out_rel = str(out.relative_to(repo_root()))
                active = True

            rows.append(
                {
                    "section": "first_frame",
                    "epsilon": EPSILON,
                    "video_id": video_id,
                    "frame_position": 0,
                    "frame_index": 0,
                    "latent_index": latent_index,
                    "active_rank": active_rank if active_rank is not None else "",
                    "active": active,
                    "halt_probability": f"{halt:.8f}" if halt is not None else "",
                    "description": description,
                    "png_path": out_rel,
                }
            )


def render_temporal_assets(rows: list[dict[str, Any]]) -> None:
    video_id = "video_76"
    for frame_position, npz_path in enumerate(sorted((attention_base() / video_id).glob("frame_*.npz"))):
        frame_index = int(npz_path.stem.split("_idx_")[-1])
        frame_src = frame_base() / video_id / f"frame_{frame_position:03d}_idx_{frame_index:06d}.png"
        frame_dst = output_base() / "media" / "video_76_sampled_frames" / f"frame_{frame_position:03d}.png"
        copy_original_frame(frame_src, frame_dst)

        for latent_index, description in TEMPORAL_TOKENS.items():
            attention, active_rank, halt = load_attention(npz_path, latent_index)
            if attention is None:
                out_rel = ""
                active = False
            else:
                out = (
                    output_base()
                    / "attention_heatmaps"
                    / "temporal"
                    / video_id
                    / f"latent_{latent_index:03d}"
                    / f"frame_{frame_position:03d}.png"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                attention_to_rgb(attention).save(out)
                out_rel = str(out.relative_to(repo_root()))
                active = True

            rows.append(
                {
                    "section": "temporal",
                    "epsilon": EPSILON,
                    "video_id": video_id,
                    "frame_position": frame_position,
                    "frame_index": frame_index,
                    "latent_index": latent_index,
                    "active_rank": active_rank if active_rank is not None else "",
                    "active": active,
                    "halt_probability": f"{halt:.8f}" if halt is not None else "",
                    "description": description,
                    "png_path": out_rel,
                }
            )


def write_manifest(rows: list[dict[str, Any]]) -> None:
    path = output_base() / "tables" / "selected_read_attention_assets.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "epsilon",
        "video_id",
        "frame_position",
        "frame_index",
        "latent_index",
        "active_rank",
        "active",
        "halt_probability",
        "description",
        "png_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict[str, Any]] = []
    render_first_frame_assets(rows)
    render_temporal_assets(rows)
    write_manifest(rows)

    print(f"Rendered {sum(1 for row in rows if row['active'])} attention heatmaps")
    print(f"Wrote manifest: {output_base() / 'tables' / 'selected_read_attention_assets.csv'}")
    print(f"Wrote sampled frames under: {output_base() / 'media' / 'video_76_sampled_frames'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
