#!/usr/bin/env python3
"""Export cam_high expert frames around the second temporal-loss peak.

This utility reads a temporal per_timestep.csv. For every expert episode it
selects the maximum requested metric only inside a progress interval, then
saves a compact cam_high-only frame window. It intentionally creates no video
files and no wrist-camera images.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

CAMERA = "observation.images.cam_high"
POLICY_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", POLICY_ROOT.parents[1])).expanduser().resolve()
DEFAULT_REPO_ID = os.environ.get("REPO_ID", "source_data_hovapi_repo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-timestep",
        type=Path,
        default=Path("loss_output/temporal_action_curves_70000_support_ego/per_timestep.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROBOTWIN_ROOT / "data" / "lerobot_data" / "huggingface" / "lerobot" / DEFAULT_REPO_ID,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "loss_output/temporal_action_curves_70000_support_ego/high_loss_frames/place_fan_second_peak_cam_high"
        ),
    )
    parser.add_argument("--task", default="place_fan")
    parser.add_argument(
        "--metric",
        choices=["mse_chunk_mean", "mse_t0", "flow_loss"],
        default="mse_chunk_mean",
        help="Metric used to select one second-peak frame per episode.",
    )
    parser.add_argument(
        "--progress-min",
        type=float,
        default=0.50,
        help="Inclusive expert-progress lower bound for the second-peak search.",
    )
    parser.add_argument(
        "--progress-max",
        type=float,
        default=0.70,
        help="Inclusive expert-progress upper bound for the second-peak search.",
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=10,
        help="Number of cam_high frames saved before the selected peak.",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=20,
        help="Number of cam_high frames saved after the selected peak.",
    )
    return parser.parse_args()


def read_loss_rows(path: Path, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["task_name"] != task:
                continue
            row["episode_index"] = int(row["episode_index"])
            row["frame_index"] = int(row["frame_index"])
            row["episode_length"] = int(row["episode_length"])
            for metric in ("expert_progress", "mse_t0", "mse_chunk_mean", "flow_loss"):
                row[metric] = float(row[metric])
            rows.append(row)
    return rows


def choose_progress_limited_peaks(
    rows: list[dict[str, Any]],
    metric: str,
    progress_min: float,
    progress_max: float,
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        progress = float(row["expert_progress"])
        if progress_min <= progress <= progress_max:
            by_episode.setdefault(int(row["episode_index"]), []).append(row)
    return [max(by_episode[index], key=lambda row: float(row[metric])) for index in sorted(by_episode)]


def episode_parquet_path(data_root: Path, episode_index: int) -> Path:
    return data_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def decode_png_cell(cell: Any) -> Image.Image:
    data = cell.get("bytes") if isinstance(cell, dict) else cell
    if data is None:
        raise ValueError("image cell has no bytes")
    return Image.open(io.BytesIO(data)).convert("RGB")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.progress_min <= args.progress_max <= 1.0:
        raise SystemExit("Require 0 <= --progress-min <= --progress-max <= 1")
    if args.pre_frames < 0 or args.post_frames < 0:
        raise SystemExit("--pre-frames and --post-frames must be non-negative")

    rows = read_loss_rows(args.per_timestep, args.task)
    if not rows:
        raise SystemExit(f"No rows found for task={args.task}: {args.per_timestep}")
    peaks = choose_progress_limited_peaks(rows, args.metric, args.progress_min, args.progress_max)
    if not peaks:
        raise SystemExit(
            f"No {args.task} samples within progress range [{args.progress_min}, {args.progress_max}]"
        )

    all_episodes = {int(row["episode_index"]) for row in rows}
    selected_episodes = {int(row["episode_index"]) for row in peaks}
    missing = sorted(all_episodes - selected_episodes)
    if missing:
        print(f"[Skip] no in-range sample for {len(missing)} episode(s): {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_by_episode_frame = {
        (int(row["episode_index"]), int(row["frame_index"])): row for row in rows
    }
    summary_rows: list[dict[str, Any]] = []

    for peak in peaks:
        episode_index = int(peak["episode_index"])
        peak_frame = int(peak["frame_index"])
        episode_length = int(peak["episode_length"])
        start = max(0, peak_frame - args.pre_frames)
        end = min(episode_length - 1, peak_frame + args.post_frames)
        parquet_path = episode_parquet_path(args.data_root, episode_index)
        if not parquet_path.is_file():
            print(f"[Skip] missing parquet: {parquet_path}")
            continue

        episode_dir = args.output_dir / (
            f"episode_{episode_index:06d}_peak_{peak_frame:04d}_p{float(peak['expert_progress']):.3f}"
        )
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        data = pd.read_parquet(parquet_path).iloc[start : end + 1]
        metric_rows: list[dict[str, Any]] = []
        for _, frame_row in data.iterrows():
            frame_index = int(frame_row["frame_index"])
            loss_row = loss_by_episode_frame.get((episode_index, frame_index), {})
            image = decode_png_cell(frame_row[CAMERA])
            image.save(frames_dir / f"frame_{frame_index:04d}.png")
            metric_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "is_peak": int(frame_index == peak_frame),
                    "expert_progress": loss_row.get("expert_progress", ""),
                    "mse_t0": loss_row.get("mse_t0", ""),
                    "mse_chunk_mean": loss_row.get("mse_chunk_mean", ""),
                    "flow_loss": loss_row.get("flow_loss", ""),
                }
            )

        write_csv(
            episode_dir / "metrics.csv",
            metric_rows,
            ["episode_index", "frame_index", "is_peak", "expert_progress", "mse_t0", "mse_chunk_mean", "flow_loss"],
        )
        info = {
            "task": args.task,
            "camera": CAMERA,
            "selection_metric": args.metric,
            "progress_range": [args.progress_min, args.progress_max],
            "episode_index": episode_index,
            "episode_length": episode_length,
            "parquet_path": str(parquet_path),
            "peak_frame": peak_frame,
            "peak_progress": float(peak["expert_progress"]),
            "peak_mse_t0": float(peak["mse_t0"]),
            "peak_mse_chunk_mean": float(peak["mse_chunk_mean"]),
            "peak_flow_loss": float(peak["flow_loss"]),
            "start_frame": start,
            "end_frame": end,
        }
        (episode_dir / "clip_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        summary_rows.append({**info, "episode_dir": str(episode_dir)})
        print(
            f"[Saved] ep={episode_index} peak={peak_frame}/{episode_length} "
            f"p={float(peak['expert_progress']):.3f} {args.metric}={float(peak[args.metric]):.6g} "
            f"range=[{start},{end}]"
        )

    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
        [
            "episode_index", "episode_length", "peak_frame", "peak_progress", "peak_mse_t0",
            "peak_mse_chunk_mean", "peak_flow_loss", "start_frame", "end_frame", "episode_dir",
        ],
    )
    (args.output_dir / "README.txt").write_text(
        "\n".join(
            [
                f"task={args.task}",
                f"camera={CAMERA}",
                f"selection_metric={args.metric}",
                f"progress_range=[{args.progress_min}, {args.progress_max}]",
                f"source_loss_csv={args.per_timestep}",
                "Each episode directory contains only cam_high PNG frames, metrics.csv, and clip_info.json.",
                "No video and no wrist-camera frames are generated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Done] output_dir={args.output_dir} episodes={len(summary_rows)}")


if __name__ == "__main__":
    main()
