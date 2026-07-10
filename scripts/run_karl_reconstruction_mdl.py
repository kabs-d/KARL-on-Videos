#!/usr/bin/env python3
"""Run task-stratified KARL reconstruction MDL on Perception Test MCQs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def shubert_root() -> Path:
    return workspace_root().parent


def default_karl_root() -> Path:
    return shubert_root() / "karl"


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
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def epsilon_tag(epsilon: float) -> str:
    return f"eps_{int(round(epsilon * 100)):03d}"


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


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


def existing_question_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        keys.add((str(row.get("row_uid")), str(row.get("epsilon_tag"))))
    return keys


def existing_video_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        keys.add((str(row.get("video_sample_uid")), str(row.get("epsilon_tag"))))
    return keys


def existing_question_video_link_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        keys.add((str(row.get("row_uid")), str(row.get("epsilon_tag"))))
    return keys


def row_cut_frame(row: dict[str, Any]) -> int | None:
    cut_frame = row.get("cut_frame")
    return int(cut_frame) if isinstance(cut_frame, int | float) else None


def video_sample_uid(row: dict[str, Any], num_frames: int, image_size: int, sampling_policy: str = "uniform") -> str:
    cut_frame = row_cut_frame(row)
    cut_tag = str(cut_frame) if cut_frame is not None else "none"
    return (
        f"{row.get('split', 'unknown')}:{row['video_id']}:cut_{cut_tag}:"
        f"{sampling_policy}{num_frames}:size_{image_size}"
    )


def group_rows_by_video_sample(
    rows: list[dict[str, Any]],
    num_frames: int,
    image_size: int,
    limit_videos: int | None = None,
) -> list[dict[str, Any]]:
    groups_by_uid: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = video_sample_uid(row, num_frames, image_size)
        if uid not in groups_by_uid:
            groups_by_uid[uid] = {
                "video_sample_uid": uid,
                "video_id": row["video_id"],
                "representative": row,
                "rows": [],
            }
        group = groups_by_uid[uid]
        if row["video_id"] != group["video_id"]:
            raise RuntimeError(f"Video sample UID collision: {uid}")
        group["rows"].append(row)
    groups = list(groups_by_uid.values())
    if limit_videos is not None:
        groups = groups[:limit_videos]
    return groups


def artifact_paths_for_video(
    output_dir: Path,
    workspace: Path,
    tag: str,
    video_id: str,
    frame_position: int,
    frame_index: int,
    include_reconstruction: bool = True,
    include_attention: bool = True,
    include_overlay: bool = True,
) -> dict[str, str | None]:
    video_dir = safe_id(video_id)
    filename = f"frame_{frame_position:03d}_idx_{frame_index:06d}"
    reconstruction_path = output_dir / "reconstructions" / tag / video_dir / f"{filename}.png"
    attention_map_path = output_dir / "attention_maps" / tag / video_dir / f"{filename}.npz"
    attention_overlay_path = output_dir / "attention_overlays" / tag / video_dir / f"{filename}.png"
    return {
        "reconstruction_path": str(reconstruction_path.relative_to(workspace)) if include_reconstruction else None,
        "attention_map_path": str(attention_map_path.relative_to(workspace)) if include_attention else None,
        "attention_overlay_path": str(attention_overlay_path.relative_to(workspace)) if include_overlay else None,
    }


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


def frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_attention_maps(attention: np.ndarray) -> np.ndarray:
    maps = np.asarray(attention, dtype=np.float32)
    if maps.size == 0:
        return maps
    flat = maps.reshape(maps.shape[0], -1)
    totals = flat.sum(axis=1, keepdims=True)
    normalized = np.divide(flat, totals, out=np.zeros_like(flat), where=totals > 0)
    return normalized.reshape(maps.shape)


def make_attention_overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    frame_img = Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")
    heat = np.asarray(heatmap, dtype=np.float32)
    if np.isfinite(heat).any() and float(np.nanmax(heat)) > 0:
        heat = heat / float(np.nanmax(heat))
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat_img = Image.fromarray(np.uint8(np.clip(heat, 0.0, 1.0) * 255), mode="L").resize(
        frame_img.size,
        Image.Resampling.BICUBIC,
    )
    red = Image.new("RGBA", frame_img.size, (255, 0, 0, 0))
    red.putalpha(heat_img.point(lambda value: int(value * alpha)))
    return Image.alpha_composite(frame_img.convert("RGBA"), red).convert("RGB")


def save_attention_overlay(
    path: Path,
    original_frame: np.ndarray,
    reconstruction_frame: np.ndarray,
    encoder_attention: np.ndarray,
    decoder_attention: np.ndarray,
    active_token_indices: np.ndarray,
    overlay_title: str,
    overlay_subtitle: str,
    epsilon_tag_value: str,
    frame_position: int,
    frame_index: int,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token_count = int(active_token_indices.shape[0])
    selected_count = min(max(top_k, 1), max(token_count, 1))
    if token_count:
        scores = encoder_attention.reshape(token_count, -1).sum(axis=1) + decoder_attention.reshape(token_count, -1).sum(axis=1)
        selected = np.argsort(scores)[::-1][:selected_count]
    else:
        selected = np.asarray([], dtype=np.int64)

    cell_w = 256
    label_h = 28
    title_h = 76
    cols = max(len(selected), 1)
    sheet = Image.new("RGB", (cols * cell_w, title_h + 2 * (label_h + cell_w)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, sheet.width, title_h], fill=(245, 245, 245))
    draw.text((8, 8), f"{overlay_title} | {epsilon_tag_value} | frame {frame_position} idx {frame_index}", fill=(0, 0, 0))
    draw.text((8, 28), overlay_subtitle[:155], fill=(0, 0, 0))
    draw.text((8, 50), "top active latent tokens: encoder over original / decoder over reconstruction", fill=(45, 45, 45))

    for col, selected_idx in enumerate(selected):
        token_id = int(active_token_indices[selected_idx])
        enc_mass = float(np.sum(encoder_attention[selected_idx]))
        dec_mass = float(np.sum(decoder_attention[selected_idx]))
        x = col * cell_w
        y0 = title_h
        draw.rectangle([x, y0, x + cell_w, y0 + label_h], fill=(250, 250, 250))
        draw.text((x + 6, y0 + 7), f"enc token {token_id} mass {enc_mass:.3f}", fill=(0, 0, 0))
        sheet.paste(make_attention_overlay(original_frame, encoder_attention[selected_idx]), (x, y0 + label_h))

        y1 = title_h + label_h + cell_w
        draw.rectangle([x, y1, x + cell_w, y1 + label_h], fill=(250, 250, 250))
        draw.text((x + 6, y1 + 7), f"dec token {token_id} mass {dec_mass:.3f}", fill=(0, 0, 0))
        sheet.paste(make_attention_overlay(reconstruction_frame, decoder_attention[selected_idx]), (x, y1 + label_h))

    if len(selected) == 0:
        draw.text((8, title_h + 8), "no active tokens found", fill=(160, 45, 45))
    sheet.save(path)


def save_karl_attention_artifacts(
    output_dir: Path,
    workspace: Path,
    artifact_id: str,
    video_sample_uid_value: str | None,
    video_id: str,
    row_uid: str | None,
    overlay_title: str,
    overlay_subtitle: str,
    tag: str,
    log: dict[str, Any],
    original_frame: np.ndarray,
    reconstruction_tensor: torch.Tensor,
    frame_position: int,
    frame_index: int,
    attention_top_k: int,
    save_overlays: bool,
    save_slot_to_latent_attention: bool = False,
    read_write_only: bool = False,
) -> tuple[str, str | None]:
    required = [
        "encoder_latent_to_input_grid_attn_16x16",
        "halt_probabilities",
    ]
    if read_write_only or save_slot_to_latent_attention:
        required.append("decoder_reconstruction_slot_to_latent_attn_16x16")
    if not read_write_only:
        required.append("decoder_latent_to_reconstruction_slot_attn_16x16")
    missing = [key for key in required if key not in log]
    if missing:
        raise KeyError(f"KARL attention log is missing keys: {missing}")

    encoder_full = to_numpy(log["encoder_latent_to_input_grid_attn_16x16"])[frame_position].astype(np.float32)
    decoder_key = (
        "decoder_reconstruction_slot_to_latent_attn_16x16"
        if read_write_only
        else "decoder_latent_to_reconstruction_slot_attn_16x16"
    )
    decoder_full = to_numpy(log[decoder_key])[frame_position].astype(np.float32)
    slot_to_latent_full = None
    if save_slot_to_latent_attention:
        slot_to_latent_full = to_numpy(log["decoder_reconstruction_slot_to_latent_attn_16x16"])[frame_position].astype(np.float32)
    halt_probabilities = to_numpy(log["halt_probabilities"])[frame_position].astype(np.float32)
    halt_threshold = float(log.get("halt_threshold", 0.75))
    active_token_indices = np.nonzero(halt_probabilities <= halt_threshold)[0].astype(np.int64)
    encoder_active = encoder_full[active_token_indices]
    decoder_active = decoder_full[active_token_indices]
    slot_to_latent_active = slot_to_latent_full[active_token_indices] if slot_to_latent_full is not None else None
    if encoder_active.ndim != 3 or encoder_active.shape[-2:] != (16, 16):
        raise RuntimeError(f"Unexpected encoder attention shape: {encoder_active.shape}")
    if decoder_active.ndim != 3 or decoder_active.shape[-2:] != (16, 16):
        raise RuntimeError(f"Unexpected decoder attention shape: {decoder_active.shape}")
    if not np.isfinite(encoder_active).all() or not np.isfinite(decoder_active).all():
        raise RuntimeError("KARL attention contains non-finite values.")
    if slot_to_latent_active is not None and not np.isfinite(slot_to_latent_active).all():
        raise RuntimeError("KARL slot-to-latent attention contains non-finite values.")

    row_dir = output_dir / "attention_maps" / tag / safe_id(artifact_id)
    map_path = row_dir / f"frame_{frame_position:03d}_idx_{frame_index:06d}.npz"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "encoder_latent_to_input_grid_attn_16x16": encoder_active,
        "active_token_indices": active_token_indices,
        "halt_probabilities": halt_probabilities,
        "epsilon_tag": np.asarray(tag),
        "video_sample_uid": np.asarray(video_sample_uid_value or ""),
        "video_id": np.asarray(video_id),
        "frame_position": np.asarray(frame_position, dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "encoder_layer": np.asarray(log.get("encoder_attention_layers", []), dtype=np.int64),
        "decoder_layer": np.asarray(log.get("decoder_attention_layers", []), dtype=np.int64),
    }
    if read_write_only:
        npz_payload.update(
            {
                "decoder_reconstruction_slot_to_latent_attn_16x16": decoder_active,
            }
        )
    else:
        npz_payload.update(
            {
                "encoder_latent_to_input_grid_attn_norm_16x16": normalize_attention_maps(encoder_active),
                "decoder_latent_to_reconstruction_slot_attn_16x16": decoder_active,
                "decoder_latent_to_reconstruction_slot_attn_norm_16x16": normalize_attention_maps(decoder_active),
                "active_halt_probabilities": halt_probabilities[active_token_indices],
                "row_uid": np.asarray(row_uid or ""),
            }
        )
        if slot_to_latent_active is not None:
            npz_payload.update(
                {
                    "decoder_reconstruction_slot_to_latent_attn_16x16": slot_to_latent_active,
                    "decoder_reconstruction_slot_to_latent_attn_norm_16x16": normalize_attention_maps(slot_to_latent_active),
                    "decoder_slot_to_latent_layer": np.asarray(log.get("decoder_slot_to_latent_attention_layers", []), dtype=np.int64),
                }
            )
    np.savez_compressed(map_path, **npz_payload)

    overlay_rel: str | None = None
    if save_overlays:
        overlay_path = output_dir / "attention_overlays" / tag / safe_id(artifact_id) / f"frame_{frame_position:03d}_idx_{frame_index:06d}.png"
        reconstruction_frame = np.asarray(tensor_to_pil(reconstruction_tensor))
        save_attention_overlay(
            overlay_path,
            original_frame,
            reconstruction_frame,
            encoder_active,
            decoder_active,
            active_token_indices,
            overlay_title,
            overlay_subtitle,
            tag,
            frame_position,
            frame_index,
            attention_top_k,
        )
        overlay_rel = str(overlay_path.relative_to(workspace))

    return str(map_path.relative_to(workspace)), overlay_rel


def tensor_values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().float().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def load_karl_model(args: argparse.Namespace, device: torch.device):
    karl_root = resolve_path(args.karl_root, shubert_root()).resolve()
    for import_root in (karl_root, karl_root / "base_tokenizers"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    ckpt_path = resolve_path(args.ckpt, karl_root)
    base_ckpt_path = resolve_path(args.base_tokenizer_ckpt, karl_root)
    if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
        raise FileNotFoundError(f"KARL checkpoint missing or empty: {ckpt_path}")
    if not base_ckpt_path.exists() or base_ckpt_path.stat().st_size == 0:
        raise FileNotFoundError(f"VQGAN base tokenizer checkpoint missing or empty: {base_ckpt_path}")

    previous_cwd = Path.cwd()
    os.chdir(karl_root)
    try:
        import kolmogorov_tokenizers

        model = kolmogorov_tokenizers.karl_small(
            base_tokenizer_args={
                "id": "vqgan",
                "is_requires_grad": False,
                "pretrained_ckpt_path": str(base_ckpt_path),
            },
            quantize_latent=True,
            train_stage="full_finetuning",
        ).to(device)
    finally:
        os.chdir(previous_cwd)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    load_result = model.load_state_dict(checkpoint["ema"], strict=False)
    model.eval()
    return model, ckpt_path, base_ckpt_path, load_result


def run_karl_batch(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    epsilon: float,
    input_token_budget: int,
    capture_attention_layers: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    stdout_buffer = io.StringIO()
    with torch.no_grad(), contextlib.redirect_stdout(stdout_buffer):
        _all_embeds, all_reconstructions, all_logs = model.encode(
            image_tensor,
            input_token_budget=[input_token_budget],
            desired_reconstruction_quality=epsilon,
            capture_attention_layers=capture_attention_layers,
        )
    if not all_reconstructions:
        raise RuntimeError("KARL encode returned no reconstructions.")
    reconstruction = all_reconstructions[-1]
    if reconstruction.shape != image_tensor.shape:
        raise RuntimeError(f"Unexpected reconstruction shape: {list(reconstruction.shape)}")
    if not torch.isfinite(reconstruction).all():
        raise RuntimeError("Reconstruction contains non-finite values.")
    return reconstruction, all_logs[-1], stdout_buffer.getvalue()


def summarize_native(question_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(question_path) if question_path.exists() else []
    by_epsilon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_epsilon_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_epsilon[row["epsilon_tag"]].append(row)
        by_epsilon_family[(row["epsilon_tag"], row["primary_task_family"])].append(row)

    def mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    summary: dict[str, Any] = {
        "total_question_epsilon_rows": len(rows),
        "by_epsilon": {},
        "by_epsilon_family": {},
    }
    for tag, tag_rows in sorted(by_epsilon.items()):
        summary["by_epsilon"][tag] = {
            "rows": len(tag_rows),
            "active_tokens_mean": mean([float(row["active_tokens_mean"]) for row in tag_rows]),
            "active_tokens_max_mean": mean([float(row["active_tokens_max"]) for row in tag_rows]),
            "reconstruction_l1_mean": mean([float(row["reconstruction_l1_mean"]) for row in tag_rows]),
        }
    for (tag, family), family_rows in sorted(by_epsilon_family.items()):
        summary["by_epsilon_family"][f"{tag}:{family}"] = {
            "epsilon_tag": tag,
            "family": family,
            "rows": len(family_rows),
            "active_tokens_mean": mean([float(row["active_tokens_mean"]) for row in family_rows]),
            "active_tokens_max_mean": mean([float(row["active_tokens_max"]) for row in family_rows]),
            "reconstruction_l1_mean": mean([float(row["reconstruction_l1_mean"]) for row in family_rows]),
        }

    lines = ["# KARL Native Reconstruction MDL", ""]
    lines.extend([
        f"- question-epsilon rows: {len(rows)}",
        "",
        "| epsilon | n | active mean | active max mean | L1 mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for tag, metrics in summary["by_epsilon"].items():
        lines.append(
            f"| {tag} | {metrics['rows']} | {metrics['active_tokens_mean']:.2f} | "
            f"{metrics['active_tokens_max_mean']:.2f} | {metrics['reconstruction_l1_mean']:.5f} |"
        )
    lines.extend(["", "## By Family", ""])
    for key, metrics in summary["by_epsilon_family"].items():
        lines.append(
            f"- {metrics['epsilon_tag']} / {metrics['family']}: n={metrics['rows']}, "
            f"active_mean={metrics['active_tokens_mean']:.2f}, "
            f"active_max_mean={metrics['active_tokens_max_mean']:.2f}, "
            f"l1_mean={metrics['reconstruction_l1_mean']:.5f}"
        )
    report_dir = output_dir / "reports"
    write_json(report_dir / "native_summary.json", summary)
    write_text(report_dir / "native_summary.md", "\n".join(lines) + "\n")
    return summary


def summarize_native_video(video_path: Path, frame_path: Path, link_path: Path, output_dir: Path) -> dict[str, Any]:
    video_rows = read_jsonl(video_path) if video_path.exists() else []
    frame_rows = read_jsonl(frame_path) if frame_path.exists() else []
    link_rows = read_jsonl(link_path) if link_path.exists() else []
    by_epsilon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in video_rows:
        by_epsilon[row["epsilon_tag"]].append(row)

    def mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    summary: dict[str, Any] = {
        "total_video_epsilon_rows": len(video_rows),
        "total_frame_epsilon_rows": len(frame_rows),
        "total_question_video_link_rows": len(link_rows),
        "unique_videos": len({row.get("video_id") for row in video_rows}),
        "unique_video_samples": len({row.get("video_sample_uid") for row in video_rows}),
        "unique_question_rows_linked": len({row.get("row_uid") for row in link_rows}),
        "by_epsilon": {},
    }
    for tag, tag_rows in sorted(by_epsilon.items()):
        summary["by_epsilon"][tag] = {
            "rows": len(tag_rows),
            "active_tokens_mean": mean([float(row["active_tokens_mean"]) for row in tag_rows]),
            "active_tokens_max_mean": mean([float(row["active_tokens_max"]) for row in tag_rows]),
            "reconstruction_l1_mean": mean([float(row["reconstruction_l1_mean"]) for row in tag_rows]),
        }

    lines = ["# KARL Native Video-Level Reconstruction MDL", ""]
    lines.extend(
        [
            f"- video-epsilon rows: {len(video_rows)}",
            f"- frame-epsilon rows: {len(frame_rows)}",
            f"- question-video link rows: {len(link_rows)}",
            f"- unique videos: {summary['unique_videos']}",
            "",
            "| epsilon | n | active mean | active max mean | L1 mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tag, metrics in summary["by_epsilon"].items():
        lines.append(
            f"| {tag} | {metrics['rows']} | {metrics['active_tokens_mean']:.2f} | "
            f"{metrics['active_tokens_max_mean']:.2f} | {metrics['reconstruction_l1_mean']:.5f} |"
        )
    report_dir = output_dir / "reports"
    write_json(report_dir / "native_video_summary.json", summary)
    write_text(report_dir / "native_video_summary.md", "\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    karl_root = default_karl_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--subset",
        type=Path,
        default=workspace / "outputs" / "curated_tasks" / "subsets" / "main_balanced_300_seed17.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=workspace / "outputs" / "karl_mdl" / "task_stratified_recon_v1")
    parser.add_argument("--karl-root", type=Path, default=karl_root)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("kolmogorov_tokenizers/pretrained_models/imagenet100/karl_small_vqgan_quantized_latents.pth"),
    )
    parser.add_argument("--base-tokenizer-ckpt", type=Path, default=Path("base_tokenizers/pretrained_models/vqgan.ckpt"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--input-token-budget", type=int, default=256)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.03, 0.05, 0.07])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--per-family-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dedupe-by-video", action="store_true")
    parser.add_argument("--no-save-reconstructions", action="store_true")
    parser.add_argument("--save-karl-attention", action="store_true")
    parser.add_argument("--attention-layers", type=int, nargs="+", default=[7])
    parser.add_argument("--attention-top-k", type=int, default=6)
    parser.add_argument("--no-attention-overlays", action="store_true")
    parser.add_argument("--save-slot-to-latent-attention", action="store_true")
    parser.add_argument("--read-write-only", action="store_true")
    return parser.parse_args()


def run_video_dedupe_mode(
    args: argparse.Namespace,
    workspace: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    run_metadata: dict[str, Any],
    start: float,
) -> int:
    frame_metrics_path = output_dir / "frame_metrics.jsonl"
    video_metrics_path = output_dir / "video_metrics.jsonl"
    link_path = output_dir / "question_video_links.jsonl"
    groups = group_rows_by_video_sample(rows, args.num_frames, args.image_size, args.limit_videos)
    selected_question_rows = [row for group in groups for row in group["rows"]]
    video_skip_keys = existing_video_keys(video_metrics_path) if args.skip_existing else set()
    link_skip_keys = existing_question_video_link_keys(link_path) if args.skip_existing else set()

    run_metadata.update(
        {
            "dedupe_by_video": True,
            "num_question_rows_selected": len(selected_question_rows),
            "num_video_samples_selected": len(groups),
            "num_unique_videos_selected": len({group["video_id"] for group in groups}),
            "question_video_link_path": str(link_path),
            "video_metrics_path": str(video_metrics_path),
        }
    )
    write_json(output_dir / "run_metadata.json", run_metadata)

    completed_video_rows = 0
    completed_link_rows = 0
    for group in tqdm(groups, desc="karl_video_reconstruction_mdl"):
        row = group["representative"]
        uid = group["video_sample_uid"]
        video_path = workspace / row["video_path"]
        cut_frame = row_cut_frame(row)
        sample_result = sample_frames_opencv(video_path, args.num_frames, cut_frame, args.image_size)
        indices = sample_result["sampled_frame_indices"]
        if cut_frame is not None and cut_frame > 0 and indices and max(indices) >= cut_frame:
            raise RuntimeError(f"cut_frame violation for {uid}: max index {max(indices)} >= {cut_frame}")
        if len(sample_result["sampled_frames"]) != len(indices):
            raise RuntimeError(f"decoded frame count mismatch for {uid}")

        image_tensor = frames_to_tensor(sample_result["sampled_frames"], torch.device("cuda:0"))
        for epsilon in args.epsilons:
            tag = epsilon_tag(epsilon)
            condition = f"karl_vqgan_{tag}_t{args.input_token_budget}_uniform{args.num_frames}"
            include_reconstruction = not args.no_save_reconstructions
            include_attention = args.save_karl_attention
            include_overlay = args.save_karl_attention and not args.no_attention_overlays
            frame_artifacts = [
                artifact_paths_for_video(
                    output_dir,
                    workspace,
                    tag,
                    row["video_id"],
                    frame_position,
                    frame_index,
                    include_reconstruction=include_reconstruction,
                    include_attention=include_attention,
                    include_overlay=include_overlay,
                )
                for frame_position, frame_index in enumerate(indices)
            ]

            if (uid, tag) not in video_skip_keys:
                reconstruction, log, encode_stdout = run_karl_batch(
                    model,
                    image_tensor,
                    epsilon,
                    args.input_token_budget,
                    capture_attention_layers=args.attention_layers if args.save_karl_attention else None,
                )
                active_counts = tensor_values(log["active_token_count"])
                halted_counts = tensor_values(log["halted_token_count"])
                l1_values = tensor_values(log["reconstruction_l1"])
                if len(active_counts) != len(indices) or len(l1_values) != len(indices):
                    raise RuntimeError(f"KARL metric/frame count mismatch for {uid} {tag}")

                frame_rows: list[dict[str, Any]] = []
                for frame_position, frame_index in enumerate(indices):
                    paths = frame_artifacts[frame_position]
                    reconstruction_path = None
                    if include_reconstruction and paths["reconstruction_path"]:
                        reconstruction_path = workspace / paths["reconstruction_path"]
                        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
                        tensor_to_pil(reconstruction[frame_position]).save(reconstruction_path)

                    attention_map_path = None
                    attention_overlay_path = None
                    if args.save_karl_attention:
                        attention_map_path, attention_overlay_path = save_karl_attention_artifacts(
                            output_dir=output_dir,
                            workspace=workspace,
                            artifact_id=row["video_id"],
                            video_sample_uid_value=uid,
                            video_id=row["video_id"],
                            row_uid=None,
                            overlay_title=row["video_id"],
                            overlay_subtitle=f"{uid} | video-level KARL artifacts",
                            tag=tag,
                            log=log,
                            original_frame=sample_result["sampled_frames"][frame_position],
                            reconstruction_tensor=reconstruction[frame_position],
                            frame_position=frame_position,
                            frame_index=frame_index,
                            attention_top_k=args.attention_top_k,
                            save_overlays=not args.no_attention_overlays,
                            save_slot_to_latent_attention=args.save_slot_to_latent_attention,
                            read_write_only=args.read_write_only,
                        )
                        paths["attention_map_path"] = attention_map_path
                        paths["attention_overlay_path"] = attention_overlay_path

                    frame_rows.append(
                        {
                            "condition": condition,
                            "epsilon": epsilon,
                            "epsilon_tag": tag,
                            "input_token_budget": args.input_token_budget,
                            "num_frames": args.num_frames,
                            "image_size": args.image_size,
                            "video_sample_uid": uid,
                            "split": row.get("split"),
                            "video_id": row["video_id"],
                            "video_path": row["video_path"],
                            "cut_frame": row.get("cut_frame"),
                            "frame_position": frame_position,
                            "frame_index": frame_index,
                            "timestamp_seconds": sample_result["sampled_timestamps_seconds"][frame_position],
                            "active_token_count": active_counts[frame_position],
                            "halted_token_count": halted_counts[frame_position],
                            "halt_threshold": float(log.get("halt_threshold", 0.75)),
                            "reconstruction_l1": l1_values[frame_position],
                            "reconstruction_path": paths["reconstruction_path"],
                            "attention_map_path": attention_map_path,
                            "attention_overlay_path": attention_overlay_path,
                        }
                    )

                video_row = {
                    "condition": condition,
                    "epsilon": epsilon,
                    "epsilon_tag": tag,
                    "input_token_budget": args.input_token_budget,
                    "num_frames": args.num_frames,
                    "image_size": args.image_size,
                    "video_sample_uid": uid,
                    "split": row.get("split"),
                    "video_id": row["video_id"],
                    "video_path": row["video_path"],
                    "cut_frame": row.get("cut_frame"),
                    "num_linked_question_rows": len(group["rows"]),
                    "video_backend": sample_result["backend"],
                    "video_frame_count": sample_result["frame_count"],
                    "video_fps": sample_result["fps"],
                    "video_duration_seconds": sample_result["duration_seconds"],
                    "sampled_frame_indices": indices,
                    "sampled_timestamps_seconds": sample_result["sampled_timestamps_seconds"],
                    "active_tokens_mean": float(np.mean(active_counts)),
                    "active_tokens_max": float(np.max(active_counts)),
                    "active_tokens_min": float(np.min(active_counts)),
                    "halted_tokens_mean": float(np.mean(halted_counts)),
                    "reconstruction_l1_mean": float(np.mean(l1_values)),
                    "reconstruction_l1_max": float(np.max(l1_values)),
                    "reconstruction_paths": [paths["reconstruction_path"] for paths in frame_artifacts],
                    "attention_map_paths": [paths["attention_map_path"] for paths in frame_artifacts],
                    "attention_overlay_paths": [paths["attention_overlay_path"] for paths in frame_artifacts],
                    "encode_stdout": encode_stdout.strip(),
                }
                append_jsonl(frame_metrics_path, frame_rows)
                append_jsonl(video_metrics_path, [video_row])
                video_skip_keys.add((uid, tag))
                completed_video_rows += 1

            link_rows: list[dict[str, Any]] = []
            for question_row in group["rows"]:
                link_key = (question_row["row_uid"], tag)
                if link_key in link_skip_keys:
                    continue
                link_rows.append(
                    {
                        "condition": condition,
                        "epsilon": epsilon,
                        "epsilon_tag": tag,
                        "input_token_budget": args.input_token_budget,
                        "num_frames": args.num_frames,
                        "image_size": args.image_size,
                        "video_sample_uid": uid,
                        "row_uid": question_row["row_uid"],
                        "split": question_row.get("split"),
                        "video_id": question_row["video_id"],
                        "question_id": question_row["question_id"],
                        "video_path": question_row["video_path"],
                        "question": question_row["question"],
                        "options": question_row["options"],
                        "answer_id": question_row["answer_id"],
                        "primary_task_family": question_row["primary_task_family"],
                        "family_memberships": question_row.get("family_memberships", []),
                        "area": question_row.get("area"),
                        "reasoning": question_row.get("reasoning"),
                        "tag": question_row.get("tag", []),
                        "cut_frame": question_row.get("cut_frame"),
                        "sampled_frame_indices": indices,
                        "sampled_timestamps_seconds": sample_result["sampled_timestamps_seconds"],
                        "reconstruction_paths": [paths["reconstruction_path"] for paths in frame_artifacts],
                        "attention_map_paths": [paths["attention_map_path"] for paths in frame_artifacts],
                        "attention_overlay_paths": [paths["attention_overlay_path"] for paths in frame_artifacts],
                    }
                )
                link_skip_keys.add(link_key)
            if link_rows:
                append_jsonl(link_path, link_rows)
                completed_link_rows += len(link_rows)

    summary = summarize_native_video(video_metrics_path, frame_metrics_path, link_path, output_dir)
    run_metadata["completed_video_epsilon_rows_this_run"] = completed_video_rows
    run_metadata["completed_question_video_link_rows_this_run"] = completed_link_rows
    run_metadata["runtime_seconds"] = time.time() - start
    run_metadata["summary"] = summary
    write_json(output_dir / "run_metadata.json", run_metadata)
    print(f"[karl-mdl] wrote video metrics: {video_metrics_path}")
    print(f"[karl-mdl] wrote frame metrics: {frame_metrics_path}")
    print(f"[karl-mdl] wrote question-video links: {link_path}")
    print(f"[karl-mdl] wrote native video summary: {output_dir / 'reports' / 'native_video_summary.md'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.read_write_only and not args.save_karl_attention:
        raise SystemExit("--read-write-only requires --save-karl-attention.")
    workspace = args.workspace.resolve()
    subset_path = resolve_path(args.subset, workspace)
    output_dir = resolve_path(args.output_dir, workspace)
    frame_metrics_path = output_dir / "frame_metrics.jsonl"
    question_metrics_path = output_dir / "question_metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run with CUDA_VISIBLE_DEVICES set to an available GPU.")
    device = torch.device("cuda:0")
    rows = select_rows(read_jsonl(subset_path), args.limit, args.per_family_limit)
    if not rows:
        raise SystemExit("No rows selected.")

    skip_keys = existing_question_keys(question_metrics_path) if args.skip_existing else set()
    model, ckpt_path, base_ckpt_path, load_result = load_karl_model(args, device)
    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": str(subset_path),
        "output_dir": str(output_dir),
        "num_rows_requested": len(rows),
        "dedupe_by_video": args.dedupe_by_video,
        "limit_videos": args.limit_videos,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "input_token_budget": args.input_token_budget,
        "epsilons": args.epsilons,
        "checkpoint_path": str(ckpt_path),
        "base_tokenizer_path": str(base_ckpt_path),
        "checkpoint_missing_keys": list(load_result.missing_keys),
        "checkpoint_unexpected_keys": list(load_result.unexpected_keys),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "save_karl_attention": args.save_karl_attention,
        "attention_layers": args.attention_layers,
        "attention_top_k": args.attention_top_k,
        "save_attention_overlays": args.save_karl_attention and not args.no_attention_overlays,
        "save_slot_to_latent_attention": args.save_slot_to_latent_attention,
        "read_write_only": args.read_write_only,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    start = time.time()
    if args.dedupe_by_video:
        return run_video_dedupe_mode(args, workspace, output_dir, rows, model, run_metadata, start)

    completed_question_rows = 0
    for row in tqdm(rows, desc="karl_reconstruction_mdl"):
        pending_epsilons = [epsilon for epsilon in args.epsilons if (row["row_uid"], epsilon_tag(epsilon)) not in skip_keys]
        if not pending_epsilons:
            continue

        video_path = workspace / row["video_path"]
        cut_frame = row.get("cut_frame")
        cut_frame = int(cut_frame) if isinstance(cut_frame, int | float) else None
        sample_result = sample_frames_opencv(video_path, args.num_frames, cut_frame, args.image_size)
        indices = sample_result["sampled_frame_indices"]
        if cut_frame is not None and cut_frame > 0 and indices and max(indices) >= cut_frame:
            raise RuntimeError(f"cut_frame violation for {row['row_uid']}: max index {max(indices)} >= {cut_frame}")
        if len(sample_result["sampled_frames"]) != len(indices):
            raise RuntimeError(f"decoded frame count mismatch for {row['row_uid']}")

        image_tensor = frames_to_tensor(sample_result["sampled_frames"], device)
        for epsilon in pending_epsilons:
            tag = epsilon_tag(epsilon)
            reconstruction, log, encode_stdout = run_karl_batch(
                model,
                image_tensor,
                epsilon,
                args.input_token_budget,
                capture_attention_layers=args.attention_layers if args.save_karl_attention else None,
            )
            active_counts = tensor_values(log["active_token_count"])
            halted_counts = tensor_values(log["halted_token_count"])
            l1_values = tensor_values(log["reconstruction_l1"])
            if len(active_counts) != len(indices) or len(l1_values) != len(indices):
                raise RuntimeError(f"KARL metric/frame count mismatch for {row['row_uid']} {tag}")

            reconstruction_paths: list[str] = []
            attention_map_paths: list[str] = []
            attention_overlay_paths: list[str | None] = []
            frame_rows: list[dict[str, Any]] = []
            row_dir = output_dir / "reconstructions" / tag / safe_id(row["row_uid"])
            for frame_position, frame_index in enumerate(indices):
                reconstruction_path = row_dir / f"frame_{frame_position:03d}_idx_{frame_index:06d}.png"
                if not args.no_save_reconstructions:
                    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
                    tensor_to_pil(reconstruction[frame_position]).save(reconstruction_path)
                    reconstruction_paths.append(str(reconstruction_path.relative_to(workspace)))

                attention_map_path = None
                attention_overlay_path = None
                if args.save_karl_attention:
                    attention_map_path, attention_overlay_path = save_karl_attention_artifacts(
                        output_dir=output_dir,
                        workspace=workspace,
                        artifact_id=row["row_uid"],
                        video_sample_uid_value=None,
                        video_id=row["video_id"],
                        row_uid=row["row_uid"],
                        overlay_title=row["row_uid"],
                        overlay_subtitle=f"{row.get('primary_task_family')} | {row.get('question', '')[:125]}",
                        tag=tag,
                        log=log,
                        original_frame=sample_result["sampled_frames"][frame_position],
                        reconstruction_tensor=reconstruction[frame_position],
                        frame_position=frame_position,
                        frame_index=frame_index,
                        attention_top_k=args.attention_top_k,
                        save_overlays=not args.no_attention_overlays,
                        save_slot_to_latent_attention=args.save_slot_to_latent_attention,
                        read_write_only=args.read_write_only,
                    )
                    attention_map_paths.append(attention_map_path)
                    attention_overlay_paths.append(attention_overlay_path)

                frame_rows.append(
                    {
                        "condition": f"karl_vqgan_{tag}_t{args.input_token_budget}_uniform{args.num_frames}",
                        "epsilon": epsilon,
                        "epsilon_tag": tag,
                        "input_token_budget": args.input_token_budget,
                        "num_frames": args.num_frames,
                        "image_size": args.image_size,
                        "row_uid": row["row_uid"],
                        "split": row.get("split"),
                        "video_id": row["video_id"],
                        "question_id": row["question_id"],
                        "primary_task_family": row["primary_task_family"],
                        "family_memberships": row.get("family_memberships", []),
                        "area": row.get("area"),
                        "reasoning": row.get("reasoning"),
                        "tag": row.get("tag", []),
                        "frame_position": frame_position,
                        "frame_index": frame_index,
                        "timestamp_seconds": sample_result["sampled_timestamps_seconds"][frame_position],
                        "active_token_count": active_counts[frame_position],
                        "halted_token_count": halted_counts[frame_position],
                        "halt_threshold": float(log.get("halt_threshold", 0.75)),
                        "reconstruction_l1": l1_values[frame_position],
                        "reconstruction_path": reconstruction_paths[-1] if reconstruction_paths else None,
                        "attention_map_path": attention_map_path,
                        "attention_overlay_path": attention_overlay_path,
                    }
                )

            question_row = {
                "condition": f"karl_vqgan_{tag}_t{args.input_token_budget}_uniform{args.num_frames}",
                "epsilon": epsilon,
                "epsilon_tag": tag,
                "input_token_budget": args.input_token_budget,
                "num_frames": args.num_frames,
                "image_size": args.image_size,
                "row_uid": row["row_uid"],
                "split": row.get("split"),
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "video_path": row["video_path"],
                "question": row["question"],
                "options": row["options"],
                "answer_id": row["answer_id"],
                "primary_task_family": row["primary_task_family"],
                "family_memberships": row.get("family_memberships", []),
                "area": row.get("area"),
                "reasoning": row.get("reasoning"),
                "tag": row.get("tag", []),
                "cut_frame": row.get("cut_frame"),
                "video_backend": sample_result["backend"],
                "video_frame_count": sample_result["frame_count"],
                "video_fps": sample_result["fps"],
                "video_duration_seconds": sample_result["duration_seconds"],
                "sampled_frame_indices": indices,
                "sampled_timestamps_seconds": sample_result["sampled_timestamps_seconds"],
                "active_tokens_mean": float(np.mean(active_counts)),
                "active_tokens_max": float(np.max(active_counts)),
                "active_tokens_min": float(np.min(active_counts)),
                "halted_tokens_mean": float(np.mean(halted_counts)),
                "reconstruction_l1_mean": float(np.mean(l1_values)),
                "reconstruction_l1_max": float(np.max(l1_values)),
                "reconstruction_paths": reconstruction_paths,
                "attention_map_paths": attention_map_paths,
                "attention_overlay_paths": attention_overlay_paths,
                "encode_stdout": encode_stdout.strip(),
            }
            append_jsonl(frame_metrics_path, frame_rows)
            append_jsonl(question_metrics_path, [question_row])
            completed_question_rows += 1

    summary = summarize_native(question_metrics_path, output_dir)
    run_metadata["completed_question_epsilon_rows_this_run"] = completed_question_rows
    run_metadata["runtime_seconds"] = time.time() - start
    run_metadata["summary"] = summary
    write_json(output_dir / "run_metadata.json", run_metadata)
    print(f"[karl-mdl] wrote frame metrics: {frame_metrics_path}")
    print(f"[karl-mdl] wrote question metrics: {question_metrics_path}")
    print(f"[karl-mdl] wrote native summary: {output_dir / 'reports' / 'native_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
