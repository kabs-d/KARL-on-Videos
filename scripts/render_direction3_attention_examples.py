"""Render Direction 3 first-frame attention examples across epsilon settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


EPSILONS = ("eps_003", "eps_005", "eps_007")
VIDEO_ID = "video_76"
FRAME_STEM = "frame_000_idx_000000"
LATENT_INDICES = (36, 38, 39, 42, 132, 158, 159)
IMAGE_SIZE = 256
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


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
    )


def output_base() -> Path:
    return (
        repo_root()
        / "results"
        / "latent_distinctiveness_v1"
        / "attention_examples"
        / VIDEO_ID
        / "frame_000"
    )


def attention_to_rgb(attention: np.ndarray) -> Image.Image:
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


def load_attention(npz_path: Path, latent_index: int) -> np.ndarray:
    with np.load(npz_path) as data:
        active = data["active_token_indices"].astype(int).tolist()
        if latent_index not in active:
            raise RuntimeError(f"latent {latent_index} is not active in {npz_path}")
        active_rank = active.index(latent_index)
        return data["encoder_latent_to_input_grid_attn_16x16"][active_rank]


def main() -> int:
    for epsilon in EPSILONS:
        npz_path = attention_base() / epsilon / VIDEO_ID / f"{FRAME_STEM}.npz"
        for latent_index in LATENT_INDICES:
            attention = load_attention(npz_path, latent_index)
            out = output_base() / f"latent_{latent_index:03d}" / f"{epsilon}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            attention_to_rgb(attention).save(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
