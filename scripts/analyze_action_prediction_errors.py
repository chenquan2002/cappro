#!/usr/bin/env python3
"""Fine-grained offline action-prediction-error analysis.

This script only reads action chunks exported by export_temporal_action_chunks.py.
It never loads a checkpoint, dataset, or model. All chunk-level metrics mask
future steps that lie beyond the end of an expert trajectory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


DEFAULT_PROGRESS_BINS = 50
DEFAULT_SIGN_EPSILON = 1e-4
CRITICAL_WINDOW = (0.6, 0.8)


@dataclass(frozen=True)
class RunSpec:
    """User-facing description of one exported action-chunk run."""

    key: str
    label: str
    success_rate: str
    directory: Path
    expected_task: str
    plot_stem: str


@dataclass
class LoadedRun:
    """Validated arrays loaded from one existing chunks.npz file."""

    spec: RunSpec
    metadata: dict[str, Any]
    pred_actions: np.ndarray
    target_actions: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    episode_length: np.ndarray
    expert_progress: np.ndarray
    action_dim_names: np.ndarray
    valid_chunk_mask: np.ndarray

    @property
    def repeat_count(self) -> int:
        return int(self.pred_actions.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.pred_actions.shape[1])

    @property
    def action_horizon(self) -> int:
        return int(self.pred_actions.shape[2])

    @property
    def action_dim(self) -> int:
        return int(self.pred_actions.shape[3])


@dataclass
class AnalysisResult:
    """All derived offline statistics for one run."""

    run: LoadedRun
    signed_error: np.ndarray
    absolute_error: np.ndarray
    squared_error: np.ndarray
    raw_chunk_mean_mse: np.ndarray
    masked_chunk_mean_mse: np.ndarray
    prediction_variance_across_r: np.ndarray
    prediction_bias_across_r: np.ndarray
    per_dof: dict[str, np.ndarray]
    progress: dict[str, dict[str, np.ndarray]]
    chunk_horizon: dict[str, np.ndarray]
    directional_consistency: dict[str, dict[str, np.ndarray]]
    sign_switch_rate: dict[str, dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=Path("loss_output/action_chunks_place_fan/support_70000"),
        help="caption-support-video place_fan export directory.",
    )
    parser.add_argument(
        "--masked-dir",
        type=Path,
        default=Path("loss_output/action_chunks_place_fan/masked_70000"),
        help="caption masked-support place_fan export directory.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("loss_output/action_chunks_place_fan/pi05_200000"),
        help="pi05 base place_fan export directory.",
    )
    parser.add_argument(
        "--grab-roller-dir",
        type=Path,
        default=None,
        help=(
            "Optional caption-support-video grab_roller export directory. "
            "It must contain chunks.npz from export_temporal_action_chunks.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("loss_output/action_error_analysis"),
        help="New analysis-only output directory. Original exports are never modified.",
    )
    parser.add_argument("--progress-bins", type=int, default=DEFAULT_PROGRESS_BINS)
    parser.add_argument(
        "--sign-epsilon",
        type=float,
        default=DEFAULT_SIGN_EPSILON,
        help=(
            "Errors with absolute value <= epsilon are treated as zero when computing sign-switch rate. "
            "Default 1e-4 is below the observed typical t0 error scale (~1e-3)."
        ),
    )
    parser.add_argument(
        "--save-full-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save masked signed/absolute/squared error tensors [R,N,50,14] to action_error_analysis.npz. "
            "Invalid padded future steps are NaN."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a prior action_error_analysis.npz in --output-dir.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _load_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata.json: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}: {error}") from error


def _load_run(spec: RunSpec) -> LoadedRun:
    directory = spec.directory.expanduser().resolve()
    npz_path = directory / "chunks.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{spec.key} requires an action-chunk export at {npz_path}. "
            "A temporal-curve CSV alone is insufficient for per-DOF/chunk analysis."
        )
    metadata = _load_metadata(directory)
    task_name = str(metadata.get("task_name", ""))
    if task_name != spec.expected_task:
        raise ValueError(
            f"{spec.key} metadata task mismatch: expected {spec.expected_task!r}, got {task_name!r}: {directory}"
        )

    required_keys = {
        "pred_actions",
        "target_actions",
        "episode_index",
        "frame_index",
        "episode_length",
        "expert_progress",
        "action_dim_labels",
    }
    with np.load(npz_path, allow_pickle=False) as arrays:
        missing = sorted(required_keys - set(arrays.files))
        if missing:
            raise KeyError(f"{npz_path} is missing required arrays: {missing}")
        pred_actions = np.asarray(arrays["pred_actions"], dtype=np.float32)
        target_actions = np.asarray(arrays["target_actions"], dtype=np.float32)
        episode_index = np.asarray(arrays["episode_index"], dtype=np.int64)
        frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
        episode_length = np.asarray(arrays["episode_length"], dtype=np.int64)
        expert_progress = np.asarray(arrays["expert_progress"], dtype=np.float32)
        action_dim_names = np.asarray(arrays["action_dim_labels"]).astype(str)

    if pred_actions.ndim != 4:
        raise ValueError(f"{npz_path}: pred_actions must be [R,N,K,D], got {pred_actions.shape}")
    repeat_count, sample_count, action_horizon, action_dim = pred_actions.shape
    expected_target_shape = (sample_count, action_horizon, action_dim)
    if target_actions.shape != expected_target_shape:
        raise ValueError(
            f"{npz_path}: target_actions expected {expected_target_shape}, got {target_actions.shape}"
        )
    for name, values in {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episode_length": episode_length,
        "expert_progress": expert_progress,
    }.items():
        if values.shape != (sample_count,):
            raise ValueError(f"{npz_path}: {name} expected shape {(sample_count,)}, got {values.shape}")
    if action_dim_names.shape != (action_dim,):
        raise ValueError(f"{npz_path}: action_dim_labels expected {(action_dim,)}, got {action_dim_names.shape}")
    if repeat_count < 1 or sample_count < 1 or action_horizon < 1 or action_dim < 1:
        raise ValueError(f"{npz_path}: invalid action shape {pred_actions.shape}")
    if not np.isfinite(pred_actions).all() or not np.isfinite(target_actions).all():
        raise ValueError(f"{npz_path}: pred_actions/target_actions contain non-finite values")
    if np.any(episode_length < 1) or np.any(frame_index < 0) or np.any(frame_index >= episode_length):
        raise ValueError(f"{npz_path}: invalid frame_index or episode_length")
    if np.any(expert_progress < -1e-6) or np.any(expert_progress > 1.0 + 1e-6):
        raise ValueError(f"{npz_path}: expert_progress is outside [0,1]")

    valid_chunk_mask = (
        np.arange(action_horizon, dtype=np.int64)[None, :] + frame_index[:, None] < episode_length[:, None]
    )
    if not np.all(valid_chunk_mask[:, 0]):
        raise AssertionError(f"{npz_path}: chunk step 0 must be valid for every selected expert frame")

    print(
        f"[Load] {spec.key}: task={task_name} R={repeat_count} N={sample_count} "
        f"K={action_horizon} D={action_dim} padded_samples={int((~valid_chunk_mask.all(axis=1)).sum())}"
    )
    return LoadedRun(
        spec=RunSpec(
            key=spec.key,
            label=spec.label,
            success_rate=spec.success_rate,
            directory=directory,
            expected_task=spec.expected_task,
            plot_stem=spec.plot_stem,
        ),
        metadata=metadata,
        pred_actions=pred_actions,
        target_actions=target_actions,
        episode_index=episode_index,
        frame_index=frame_index,
        episode_length=episode_length,
        expert_progress=expert_progress,
        action_dim_names=action_dim_names,
        valid_chunk_mask=valid_chunk_mask,
    )


def _validate_comparable_place_fan_runs(runs: list[LoadedRun]) -> None:
    first = runs[0]
    for other in runs[1:]:
        if first.pred_actions.shape[1:] != other.pred_actions.shape[1:]:
            raise ValueError(
                f"Place-fan shape mismatch: {first.spec.key}={first.pred_actions.shape}, "
                f"{other.spec.key}={other.pred_actions.shape}"
            )
        for name in (
            "episode_index",
            "frame_index",
            "episode_length",
            "expert_progress",
            "target_actions",
            "action_dim_names",
        ):
            if not np.array_equal(getattr(first, name), getattr(other, name)):
                raise ValueError(f"Place-fan inputs differ for {name}: {first.spec.key} vs {other.spec.key}")


def _progress_bin_indices(progress: np.ndarray, bins: int) -> np.ndarray:
    clipped = np.clip(progress, 0.0, 1.0)
    return np.minimum((clipped * bins).astype(np.int64), bins - 1)


def _mean_std_sem(values: np.ndarray, axes: tuple[int, ...]) -> dict[str, np.ndarray]:
    """NaN-aware mean/std/SEM and finite count along the specified axes."""

    finite_count = np.sum(np.isfinite(values), axis=axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=axes)
        std = np.nanstd(values, axis=axes, ddof=1)
    std = np.where(finite_count > 1, std, np.where(finite_count == 1, 0.0, np.nan))
    sem = np.where(finite_count > 1, std / np.sqrt(finite_count), np.where(finite_count == 1, 0.0, np.nan))
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "sem": np.asarray(sem, dtype=np.float64),
        "count": np.asarray(finite_count, dtype=np.int64),
    }


def _episode_progress_stats(
    values: np.ndarray,
    *,
    episode_index: np.ndarray,
    progress_bins: np.ndarray,
    bin_count: int,
    postprocess_episode_mean=None,
) -> dict[str, np.ndarray]:
    """Aggregate [R,N,D] values by episode first, then by common progress bin.

    Each episode gets equal weight in a bin, rather than allowing longer
    trajectories to dominate the uncertainty estimate. R is retained as a
    separate independent sampling axis until this final statistical reduction.
    """

    if values.ndim != 3:
        raise ValueError(f"Expected [R,N,D], got {values.shape}")
    repeat_count, _, action_dim = values.shape
    episode_ids = np.unique(episode_index)
    per_episode = np.full((repeat_count, len(episode_ids), bin_count, action_dim), np.nan, dtype=np.float64)

    for episode_position, episode_id in enumerate(episode_ids):
        trajectory_indices = np.flatnonzero(episode_index == episode_id)
        for bin_index in np.unique(progress_bins[trajectory_indices]):
            in_bin = trajectory_indices[progress_bins[trajectory_indices] == bin_index]
            per_episode[:, episode_position, int(bin_index), :] = np.mean(values[:, in_bin, :], axis=1)

    if postprocess_episode_mean is not None:
        per_episode = postprocess_episode_mean(per_episode)
    stats = _mean_std_sem(per_episode, axes=(0, 1))
    stats["per_episode_repeat"] = per_episode
    stats["episode_ids"] = episode_ids.astype(np.int64)
    return stats


def _masked_chunk_per_sample(values: np.ndarray, valid_chunk_mask: np.ndarray) -> np.ndarray:
    """Mean [R,N,K,D] values over valid chunk steps, separately for each DOF."""

    mask = valid_chunk_mask[None, :, :, None]
    denominators = valid_chunk_mask.sum(axis=1, dtype=np.int64)[None, :, None]
    totals = np.where(mask, values, 0.0).sum(axis=2, dtype=np.float64)
    return totals / denominators


def _chunk_horizon_stats(
    signed_error: np.ndarray,
    absolute_error: np.ndarray,
    squared_error: np.ndarray,
    valid_chunk_mask: np.ndarray,
    progress_bins: np.ndarray,
    bin_count: int,
) -> dict[str, np.ndarray]:
    """Compute metric(progress-bin, chunk-step, DOF) with tail padding removed."""

    repeat_count, _, action_horizon, action_dim = signed_error.shape
    shape = (bin_count, action_horizon, action_dim)
    mae = np.full(shape, np.nan, dtype=np.float64)
    mse = np.full(shape, np.nan, dtype=np.float64)
    signed_bias = np.full(shape, np.nan, dtype=np.float64)
    mae_aggregate = np.full((bin_count, action_horizon), np.nan, dtype=np.float64)
    mse_aggregate = np.full((bin_count, action_horizon), np.nan, dtype=np.float64)
    valid_sample_count = np.zeros((bin_count, action_horizon), dtype=np.int64)

    for bin_index in range(bin_count):
        indices = np.flatnonzero(progress_bins == bin_index)
        if len(indices) == 0:
            continue
        local_mask = valid_chunk_mask[indices]
        valid_sample_count[bin_index] = local_mask.sum(axis=0)
        broadcast_mask = local_mask[None, :, :, None]
        denominator = (repeat_count * valid_sample_count[bin_index])[:, None]
        valid_steps = valid_sample_count[bin_index] > 0

        for source, destination in (
            (absolute_error, mae),
            (squared_error, mse),
            (signed_error, signed_bias),
        ):
            totals = np.where(broadcast_mask, source[:, indices, :, :], 0.0).sum(axis=(0, 1), dtype=np.float64)
            destination[bin_index, valid_steps] = totals[valid_steps] / denominator[valid_steps]

        aggregate_denominator = repeat_count * valid_sample_count[bin_index] * action_dim
        for source, destination in (
            (absolute_error, mae_aggregate),
            (squared_error, mse_aggregate),
        ):
            totals = np.where(broadcast_mask, source[:, indices, :, :], 0.0).sum(axis=(0, 1, 3), dtype=np.float64)
            destination[bin_index, valid_steps] = (
                totals[valid_steps] / aggregate_denominator[valid_steps]
            )

    return {
        "mae": mae,
        "mse": mse,
        "signed_bias": signed_bias,
        "mae_aggregate": mae_aggregate,
        "mse_aggregate": mse_aggregate,
        "valid_sample_count": valid_sample_count,
    }


def _trajectory_directional_metrics(
    signed_t0_error: np.ndarray,
    *,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    expert_progress: np.ndarray,
    sign_epsilon: float,
    progress_window: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-repeat/per-episode DC, SSR, and the episode id order.

    The temporal metrics are intentionally defined from chunk[0] only. This
    avoids counting overlapping future predictions many times and avoids any
    trajectory-tail padding issue.
    """

    repeat_count, _, action_dim = signed_t0_error.shape
    episode_ids = np.unique(episode_index)
    dc = np.full((repeat_count, len(episode_ids), action_dim), np.nan, dtype=np.float64)
    ssr = np.full((repeat_count, len(episode_ids), action_dim), np.nan, dtype=np.float64)

    for episode_position, episode_id in enumerate(episode_ids):
        indices = np.flatnonzero(episode_index == episode_id)
        indices = indices[np.argsort(frame_index[indices], kind="stable")]
        if progress_window is not None:
            lower, upper = progress_window
            indices = indices[
                (expert_progress[indices] >= lower) & (expert_progress[indices] <= upper)
            ]
        if len(indices) == 0:
            continue

        trajectory_error = signed_t0_error[:, indices, :]
        dc[:, episode_position, :] = np.abs(trajectory_error.sum(axis=1)) / (
            np.abs(trajectory_error).sum(axis=1) + 1e-12
        )

        if len(indices) < 2:
            continue
        signs = np.where(
            trajectory_error > sign_epsilon,
            1,
            np.where(trajectory_error < -sign_epsilon, -1, 0),
        )
        valid_pairs = (signs[:, 1:, :] != 0) & (signs[:, :-1, :] != 0)
        pair_count = valid_pairs.sum(axis=1)
        switches = ((signs[:, 1:, :] != signs[:, :-1, :]) & valid_pairs).sum(axis=1)
        np.divide(
            switches,
            pair_count,
            out=ssr[:, episode_position, :],
            where=pair_count > 0,
        )

    return dc, ssr, episode_ids.astype(np.int64)


def _analyze_run(run: LoadedRun, *, progress_bin_count: int, sign_epsilon: float) -> AnalysisResult:
    """Compute all offline metrics from the full [R,N,K,D] action error tensor."""

    raw_signed_error = run.pred_actions - run.target_actions[None, ...]
    raw_absolute_error = np.abs(raw_signed_error)
    raw_squared_error = np.square(raw_signed_error)
    mask = run.valid_chunk_mask[None, :, :, None]

    # Invalid future steps are NaN in saved errors. This makes accidental use of
    # padded target actions visible and forces downstream nan-aware reductions.
    signed_error = np.where(mask, raw_signed_error, np.nan).astype(np.float32)
    absolute_error = np.where(mask, raw_absolute_error, np.nan).astype(np.float32)
    squared_error = np.where(mask, raw_squared_error, np.nan).astype(np.float32)

    raw_chunk_mean_mse = raw_squared_error.mean(axis=(-2, -1), dtype=np.float64)
    valid_action_count = run.valid_chunk_mask.sum(axis=1, dtype=np.int64) * run.action_dim
    masked_chunk_mean_mse = (
        np.where(mask, raw_squared_error, 0.0).sum(axis=(-2, -1), dtype=np.float64)
        / valid_action_count[None, :]
    )

    prediction_variance_across_r = np.var(run.pred_actions, axis=0, dtype=np.float64)
    prediction_bias_across_r = np.mean(raw_signed_error, axis=0, dtype=np.float64)
    prediction_variance_across_r = np.where(
        run.valid_chunk_mask[:, :, None], prediction_variance_across_r, np.nan
    ).astype(np.float32)
    prediction_bias_across_r = np.where(
        run.valid_chunk_mask[:, :, None], prediction_bias_across_r, np.nan
    ).astype(np.float32)

    progress_bins = _progress_bin_indices(run.expert_progress, progress_bin_count)
    t0_signed_error = raw_signed_error[:, :, 0, :]
    t0_absolute_error = raw_absolute_error[:, :, 0, :]
    t0_squared_error = raw_squared_error[:, :, 0, :]

    full_chunk_signed_per_sample = _masked_chunk_per_sample(raw_signed_error, run.valid_chunk_mask)
    full_chunk_absolute_per_sample = _masked_chunk_per_sample(raw_absolute_error, run.valid_chunk_mask)
    full_chunk_squared_per_sample = _masked_chunk_per_sample(raw_squared_error, run.valid_chunk_mask)

    per_dof = {
        "t0_mae": np.mean(t0_absolute_error, axis=(0, 1), dtype=np.float64),
        "t0_rmse": np.sqrt(np.mean(t0_squared_error, axis=(0, 1), dtype=np.float64)),
        "t0_signed_bias": np.mean(t0_signed_error, axis=(0, 1), dtype=np.float64),
        "full_chunk_mae": np.nanmean(absolute_error, axis=(0, 1, 2), dtype=np.float64),
        "full_chunk_rmse": np.sqrt(np.nanmean(squared_error, axis=(0, 1, 2), dtype=np.float64)),
        "full_chunk_signed_bias": np.nanmean(signed_error, axis=(0, 1, 2), dtype=np.float64),
        "raw_chunk_mean_mse": np.mean(raw_chunk_mean_mse, axis=(0, 1), dtype=np.float64),
        "masked_chunk_mean_mse": np.mean(masked_chunk_mean_mse, axis=(0, 1), dtype=np.float64),
    }

    progress = {
        "t0_mae": _episode_progress_stats(
            t0_absolute_error,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
        ),
        "t0_rmse": _episode_progress_stats(
            t0_squared_error,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
            postprocess_episode_mean=np.sqrt,
        ),
        "t0_signed_bias": _episode_progress_stats(
            t0_signed_error,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
        ),
        "full_chunk_mae": _episode_progress_stats(
            full_chunk_absolute_per_sample,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
        ),
        "full_chunk_rmse": _episode_progress_stats(
            full_chunk_squared_per_sample,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
            postprocess_episode_mean=np.sqrt,
        ),
        "full_chunk_signed_bias": _episode_progress_stats(
            full_chunk_signed_per_sample,
            episode_index=run.episode_index,
            progress_bins=progress_bins,
            bin_count=progress_bin_count,
        ),
    }

    chunk_horizon = _chunk_horizon_stats(
        raw_signed_error,
        raw_absolute_error,
        raw_squared_error,
        run.valid_chunk_mask,
        progress_bins,
        progress_bin_count,
    )

    dc_full, ssr_full, episode_ids = _trajectory_directional_metrics(
        t0_signed_error,
        episode_index=run.episode_index,
        frame_index=run.frame_index,
        expert_progress=run.expert_progress,
        sign_epsilon=sign_epsilon,
        progress_window=None,
    )
    dc_critical, ssr_critical, critical_episode_ids = _trajectory_directional_metrics(
        t0_signed_error,
        episode_index=run.episode_index,
        frame_index=run.frame_index,
        expert_progress=run.expert_progress,
        sign_epsilon=sign_epsilon,
        progress_window=CRITICAL_WINDOW,
    )
    if not np.array_equal(episode_ids, critical_episode_ids):
        raise AssertionError("Episode ordering changed between full and critical-window temporal metrics")

    directional_consistency = {
        "episode_ids": episode_ids,
        "full_per_episode_repeat": dc_full,
        "critical_per_episode_repeat": dc_critical,
        "full": _mean_std_sem(dc_full, axes=(0, 1)),
        "critical": _mean_std_sem(dc_critical, axes=(0, 1)),
    }
    sign_switch_rate = {
        "episode_ids": episode_ids,
        "full_per_episode_repeat": ssr_full,
        "critical_per_episode_repeat": ssr_critical,
        "full": _mean_std_sem(ssr_full, axes=(0, 1)),
        "critical": _mean_std_sem(ssr_critical, axes=(0, 1)),
    }

    return AnalysisResult(
        run=run,
        signed_error=signed_error,
        absolute_error=absolute_error,
        squared_error=squared_error,
        raw_chunk_mean_mse=raw_chunk_mean_mse,
        masked_chunk_mean_mse=masked_chunk_mean_mse,
        prediction_variance_across_r=prediction_variance_across_r,
        prediction_bias_across_r=prediction_bias_across_r,
        per_dof=per_dof,
        progress=progress,
        chunk_horizon=chunk_horizon,
        directional_consistency=directional_consistency,
        sign_switch_rate=sign_switch_rate,
    )


def _plot_per_dof_progress(
    results: list[AnalysisResult],
    *,
    metric: str,
    output_stem: Path,
    title: str,
    y_label: str,
    draw_zero: bool,
    progress_centers: np.ndarray,
) -> None:
    action_names = results[0].run.action_dim_names
    figure, axes = plt.subplots(2, 7, figsize=(23, 6.7), sharex=True)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
    lines = []
    labels = []

    for action_index, axis in enumerate(axes.flat):
        for result, color in zip(results, colors, strict=False):
            stats = result.progress[metric]
            mean = stats["mean"][:, action_index]
            sem = stats["sem"][:, action_index]
            valid = np.isfinite(mean)
            line = axis.plot(progress_centers[valid], mean[valid], color=color, linewidth=1.8)[0]
            if np.any(valid):
                lower = mean[valid] - sem[valid]
                upper = mean[valid] + sem[valid]
                axis.fill_between(progress_centers[valid], lower, upper, color=color, alpha=0.17, linewidth=0)
            if action_index == 0:
                lines.append(line)
                labels.append(f"{result.run.spec.label} ({result.run.spec.success_rate})")

        if draw_zero:
            axis.axhline(0.0, color="black", linewidth=0.75, linestyle="--", alpha=0.65)
        axis.set_title(str(action_names[action_index]), fontsize=10)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlim(0.0, 1.0)
        if action_index % 7 == 0:
            axis.set_ylabel(y_label)
        if action_index >= 7:
            axis.set_xlabel("Expert trajectory progress")

    figure.suptitle(title, y=1.03, fontsize=14)
    figure.legend(lines, labels, loc="upper center", ncol=len(results), frameon=False, bbox_to_anchor=(0.5, 0.99))
    figure.tight_layout()
    _save_figure(figure, output_stem)


def _plot_temporal_heatmap(
    results: list[AnalysisResult],
    *,
    metric_name: str,
    output_stem: Path,
    title: str,
) -> None:
    action_names = results[0].run.action_dim_names
    figure, axes = plt.subplots(1, 2, figsize=(18, max(3.3, 1.0 + 0.75 * len(results))), squeeze=False)
    axes_flat = axes.flat
    metric_attr = "directional_consistency" if metric_name == "dc" else "sign_switch_rate"
    titles = ("Full trajectory", "Progress 0.6-0.8")

    for axis, statistic_key, panel_title in zip(axes_flat, ("full", "critical"), titles, strict=True):
        values = np.stack(
            [getattr(result, metric_attr)[statistic_key]["mean"] for result in results],
            axis=0,
        )
        image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.set_xticks(np.arange(len(action_names)))
        axis.set_xticklabels(action_names, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(results)))
        axis.set_yticklabels([f"{result.run.spec.label}\n({result.run.spec.success_rate})" for result in results])
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    text_color = "white" if value > 0.55 else "black"
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(title, y=1.02, fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output_stem)


def _plot_chunk_horizon_heatmaps(
    results: list[AnalysisResult],
    *,
    output_directory: Path,
) -> None:
    all_values = np.concatenate(
        [
            result.chunk_horizon["mae_aggregate"][np.isfinite(result.chunk_horizon["mae_aggregate"])]
            for result in results
        ]
    )
    if len(all_values) == 0:
        raise ValueError("No finite chunk-horizon MAE values available for plotting")
    color_max = float(np.quantile(all_values, 0.995))
    color_max = max(color_max, float(np.nanmax(all_values)) * 0.05, 1e-12)

    for result in results:
        values = result.chunk_horizon["mae_aggregate"].T
        masked_values = np.ma.masked_invalid(values)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="#d9d9d9")
        figure, axis = plt.subplots(figsize=(8.5, 7.0))
        image = axis.imshow(
            masked_values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=color_max,
            extent=(0.0, 1.0, result.run.action_horizon - 1, 0.0),
        )
        axis.set_xlabel("Expert trajectory progress")
        axis.set_ylabel("Action chunk step")
        axis.set_yticks(np.arange(0, result.run.action_horizon, 5))
        axis.set_title(f"{result.run.spec.label} ({result.run.spec.success_rate}): chunk-horizon MAE")
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("MAE (shared 99.5th-percentile scale)")
        figure.tight_layout()
        _save_figure(figure, output_directory / f"{result.run.spec.plot_stem}_chunk_horizon_mae")


def _build_npz_arrays(
    results: list[AnalysisResult],
    *,
    progress_edges: np.ndarray,
    progress_centers: np.ndarray,
    sign_epsilon: float,
    save_full_errors: bool,
) -> dict[str, np.ndarray]:
    """Build one self-contained numerical result file without touching source npz files."""

    arrays: dict[str, np.ndarray] = {
        "model_keys": np.asarray([result.run.spec.key for result in results]),
        "model_labels": np.asarray([result.run.spec.label for result in results]),
        "success_rates": np.asarray([result.run.spec.success_rate for result in results]),
        "action_dim_names": np.asarray(results[0].run.action_dim_names),
        "progress_bin_edges": progress_edges,
        "progress_bin_centers": progress_centers,
        "critical_progress_window": np.asarray(CRITICAL_WINDOW, dtype=np.float64),
        "sign_epsilon": np.asarray(sign_epsilon, dtype=np.float64),
        "temporal_metrics_chunk_index": np.asarray(0, dtype=np.int64),
    }

    for result in results:
        key = result.run.spec.key
        run = result.run
        arrays.update(
            {
                f"{key}__episode_index": run.episode_index,
                f"{key}__frame_index": run.frame_index,
                f"{key}__episode_length": run.episode_length,
                f"{key}__expert_progress": run.expert_progress,
                f"{key}__valid_chunk_mask": run.valid_chunk_mask,
                f"{key}__raw_chunk_mean_mse": result.raw_chunk_mean_mse,
                f"{key}__masked_chunk_mean_mse": result.masked_chunk_mean_mse,
                f"{key}__prediction_variance_across_r": result.prediction_variance_across_r,
                f"{key}__prediction_bias_across_r": result.prediction_bias_across_r,
            }
        )
        if save_full_errors:
            arrays.update(
                {
                    f"{key}__signed_error": result.signed_error,
                    f"{key}__absolute_error": result.absolute_error,
                    f"{key}__squared_error": result.squared_error,
                }
            )

        for metric_name, values in result.per_dof.items():
            arrays[f"{key}__per_dof_{metric_name}"] = np.asarray(values)

        for metric_name, stats in result.progress.items():
            for statistic_name, values in stats.items():
                arrays[f"{key}__progress_{metric_name}_{statistic_name}"] = np.asarray(values)

        for metric_name, values in result.chunk_horizon.items():
            arrays[f"{key}__chunk_horizon_{metric_name}"] = np.asarray(values)

        for container_name, container in (
            ("directional_consistency", result.directional_consistency),
            ("sign_switch_rate", result.sign_switch_rate),
        ):
            for statistic_name, values in container.items():
                if isinstance(values, dict):
                    for nested_name, nested_values in values.items():
                        arrays[f"{key}__{container_name}_{statistic_name}_{nested_name}"] = np.asarray(nested_values)
                else:
                    arrays[f"{key}__{container_name}_{statistic_name}"] = np.asarray(values)
    return arrays


def _build_per_dof_summary_rows(results: list[AnalysisResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for action_index, action_name in enumerate(result.run.action_dim_names):
            row: dict[str, Any] = {
                "model_key": result.run.spec.key,
                "model_label": result.run.spec.label,
                "success_rate": result.run.spec.success_rate,
                "task_name": result.run.spec.expected_task,
                "action_index": action_index,
                "action_name": str(action_name),
            }
            for metric_name, values in result.per_dof.items():
                row[metric_name] = float(values[action_index]) if np.ndim(values) else float(values)
            for name, container in (
                ("directional_consistency", result.directional_consistency),
                ("sign_switch_rate", result.sign_switch_rate),
            ):
                for window in ("full", "critical"):
                    for statistic in ("mean", "std", "sem", "count"):
                        row[f"{name}_{window}_{statistic}"] = float(
                            container[window][statistic][action_index]
                        )
            rows.append(row)
    return rows


def _build_progress_rows(
    results: list[AnalysisResult],
    *,
    progress_edges: np.ndarray,
    progress_centers: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for metric_name, stats in result.progress.items():
            for bin_index, (start, end, center) in enumerate(
                zip(progress_edges[:-1], progress_edges[1:], progress_centers, strict=True)
            ):
                for action_index, action_name in enumerate(result.run.action_dim_names):
                    rows.append(
                        {
                            "model_key": result.run.spec.key,
                            "task_name": result.run.spec.expected_task,
                            "metric": metric_name,
                            "progress_bin_index": bin_index,
                            "progress_start": float(start),
                            "progress_end": float(end),
                            "progress_center": float(center),
                            "action_index": action_index,
                            "action_name": str(action_name),
                            "mean": float(stats["mean"][bin_index, action_index]),
                            "std": float(stats["std"][bin_index, action_index]),
                            "sem": float(stats["sem"][bin_index, action_index]),
                            "count": int(stats["count"][bin_index, action_index]),
                        }
                    )
    return rows


def _build_run_summary_rows(results: list[AnalysisResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        run = result.run
        rows.append(
            {
                "model_key": run.spec.key,
                "model_label": run.spec.label,
                "success_rate": run.spec.success_rate,
                "task_name": run.spec.expected_task,
                "source_dir": str(run.spec.directory),
                "repeat_count": run.repeat_count,
                "sample_count": run.sample_count,
                "episode_count": int(len(np.unique(run.episode_index))),
                "action_horizon": run.action_horizon,
                "action_dim": run.action_dim,
                "samples_with_padding": int((~run.valid_chunk_mask.all(axis=1)).sum()),
                "valid_chunk_entries": int(run.valid_chunk_mask.sum()),
                "total_chunk_entries": int(run.valid_chunk_mask.size),
                "raw_chunk_mean_mse": float(np.mean(result.raw_chunk_mean_mse)),
                "masked_chunk_mean_mse": float(np.mean(result.masked_chunk_mean_mse)),
                "padding_mse_delta_raw_minus_masked": float(
                    np.mean(result.raw_chunk_mean_mse) - np.mean(result.masked_chunk_mean_mse)
                ),
            }
        )
    return rows


def _analysis_metadata(
    results: list[AnalysisResult],
    *,
    args: argparse.Namespace,
    progress_edges: np.ndarray,
    grab_roller_missing: bool,
) -> dict[str, Any]:
    return {
        "mode": "offline_action_prediction_error_analysis",
        "source_model_inference": False,
        "source_npz_modified": False,
        "progress_bin_count": int(args.progress_bins),
        "progress_bin_edges": progress_edges.tolist(),
        "progress_binning": (
            "expert_progress is assigned to 50 fixed [0,1] bins. For per-DOF curves, "
            "frame-level values are averaged within each episode/bin first; episode x repeat values "
            "are then used for mean/std/SEM."
        ),
        "valid_chunk_mask": "arange(chunk_horizon) + frame_index < episode_length",
        "padding_policy": (
            "All chunk-level reductions ignore valid_chunk_mask=False entries. Saved full error "
            "arrays store NaN at invalid padded chunk steps."
        ),
        "temporal_metrics": (
            "Directional consistency and sign-switch rate use chunk[0] error only, are computed "
            "within each trajectory/repeat first, then aggregated across trajectories/repeats."
        ),
        "critical_progress_window": list(CRITICAL_WINDOW),
        "sign_switch_epsilon": float(args.sign_epsilon),
        "sign_switch_definition": (
            "sign=+1 for error>epsilon, -1 for error<-epsilon, else 0. "
            "Only adjacent pairs where both signs are nonzero enter the denominator."
        ),
        "r_dimension": (
            "Raw signed/absolute/squared error tensors retain R. Prediction variance and bias are "
            "computed across R. Aggregate statistics reduce R only at the final statistic stage."
        ),
        "save_full_errors": bool(args.save_full_errors),
        "grab_roller_missing": grab_roller_missing,
        "runs": [
            {
                "key": result.run.spec.key,
                "label": result.run.spec.label,
                "success_rate": result.run.spec.success_rate,
                "task_name": result.run.spec.expected_task,
                "source_dir": str(result.run.spec.directory),
                "repeat_count": result.run.repeat_count,
                "sample_count": result.run.sample_count,
                "metadata": result.run.metadata,
            }
            for result in results
        ],
    }


def _plot_all(
    place_fan_results: list[AnalysisResult],
    *,
    grab_roller_result: AnalysisResult | None,
    output_dir: Path,
    progress_centers: np.ndarray,
) -> None:
    plots_dir = output_dir / "plots"
    _plot_per_dof_progress(
        place_fan_results,
        metric="t0_mae",
        output_stem=plots_dir / "place_fan_per_dof_mae",
        title="place_fan: per-DOF chunk[0] MAE",
        y_label="MAE",
        draw_zero=False,
        progress_centers=progress_centers,
    )
    _plot_per_dof_progress(
        place_fan_results,
        metric="t0_signed_bias",
        output_stem=plots_dir / "place_fan_per_dof_signed_bias",
        title="place_fan: per-DOF chunk[0] signed bias",
        y_label="Mean prediction - target",
        draw_zero=True,
        progress_centers=progress_centers,
    )
    _plot_temporal_heatmap(
        place_fan_results,
        metric_name="dc",
        output_stem=plots_dir / "place_fan_directional_consistency",
        title="place_fan: directional consistency of chunk[0] signed error",
    )
    _plot_temporal_heatmap(
        place_fan_results,
        metric_name="ssr",
        output_stem=plots_dir / "place_fan_sign_switch_rate",
        title="place_fan: sign-switch rate of chunk[0] signed error",
    )
    _plot_chunk_horizon_heatmaps(
        place_fan_results,
        output_directory=plots_dir,
    )

    if grab_roller_result is None:
        return
    _plot_per_dof_progress(
        [grab_roller_result],
        metric="t0_mae",
        output_stem=plots_dir / "grab_roller_per_dof_mae",
        title="grab_roller: per-DOF chunk[0] MAE",
        y_label="MAE",
        draw_zero=False,
        progress_centers=progress_centers,
    )
    _plot_per_dof_progress(
        [grab_roller_result],
        metric="t0_signed_bias",
        output_stem=plots_dir / "grab_roller_per_dof_signed_bias",
        title="grab_roller: per-DOF chunk[0] signed bias",
        y_label="Mean prediction - target",
        draw_zero=True,
        progress_centers=progress_centers,
    )
    _plot_temporal_heatmap(
        [grab_roller_result],
        metric_name="dc",
        output_stem=plots_dir / "grab_roller_directional_consistency",
        title="grab_roller: directional consistency of chunk[0] signed error",
    )
    _plot_temporal_heatmap(
        [grab_roller_result],
        metric_name="ssr",
        output_stem=plots_dir / "grab_roller_sign_switch_rate",
        title="grab_roller: sign-switch rate of chunk[0] signed error",
    )
    _plot_chunk_horizon_heatmaps(
        [grab_roller_result],
        output_directory=plots_dir,
    )


def analyze(args: argparse.Namespace) -> None:
    if args.progress_bins < 1:
        raise ValueError("--progress-bins must be >= 1")
    if args.sign_epsilon < 0.0:
        raise ValueError("--sign-epsilon must be >= 0")
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "action_error_analysis.npz"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing analysis: {result_path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    place_specs = [
        RunSpec(
            key="support_video",
            label="Caption + support video",
            success_rate="4%",
            directory=args.support_dir,
            expected_task="place_fan",
            plot_stem="place_fan_support",
        ),
        RunSpec(
            key="masked_support",
            label="Caption, support masked",
            success_rate="40%",
            directory=args.masked_dir,
            expected_task="place_fan",
            plot_stem="place_fan_masked",
        ),
        RunSpec(
            key="pi05_base",
            label="pi05 base",
            success_rate="60%",
            directory=args.base_dir,
            expected_task="place_fan",
            plot_stem="place_fan_base",
        ),
    ]
    place_fan_runs = [_load_run(spec) for spec in place_specs]
    _validate_comparable_place_fan_runs(place_fan_runs)

    grab_roller_run: LoadedRun | None = None
    grab_roller_missing = args.grab_roller_dir is None
    if args.grab_roller_dir is not None:
        grab_roller_run = _load_run(
            RunSpec(
                key="grab_roller_support_video",
                label="Caption + support video",
                success_rate="≈100%",
                directory=args.grab_roller_dir,
                expected_task="grab_roller",
                plot_stem="grab_roller",
            )
        )
        if not np.array_equal(place_fan_runs[0].action_dim_names, grab_roller_run.action_dim_names):
            raise ValueError("grab_roller action_dim_labels differ from place_fan")

    print("[Analyze] computing full error tensors and offline statistics")
    place_fan_results = [
        _analyze_run(run, progress_bin_count=args.progress_bins, sign_epsilon=args.sign_epsilon)
        for run in place_fan_runs
    ]
    grab_roller_result = (
        None
        if grab_roller_run is None
        else _analyze_run(grab_roller_run, progress_bin_count=args.progress_bins, sign_epsilon=args.sign_epsilon)
    )
    all_results = [*place_fan_results, *(() if grab_roller_result is None else (grab_roller_result,))]

    progress_edges = np.linspace(0.0, 1.0, args.progress_bins + 1, dtype=np.float64)
    progress_centers = (progress_edges[:-1] + progress_edges[1:]) / 2.0
    arrays = _build_npz_arrays(
        all_results,
        progress_edges=progress_edges,
        progress_centers=progress_centers,
        sign_epsilon=args.sign_epsilon,
        save_full_errors=args.save_full_errors,
    )
    np.savez_compressed(result_path, **arrays)
    _write_csv(output_dir / "per_dof_summary.csv", _build_per_dof_summary_rows(all_results))
    _write_csv(
        output_dir / "progress_per_dof.csv",
        _build_progress_rows(all_results, progress_edges=progress_edges, progress_centers=progress_centers),
    )
    _write_csv(output_dir / "run_summary.csv", _build_run_summary_rows(all_results))
    metadata = _analysis_metadata(
        all_results,
        args=args,
        progress_edges=progress_edges,
        grab_roller_missing=grab_roller_missing,
    )
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("[Plot] writing offline analysis figures")
    _plot_all(
        place_fan_results,
        grab_roller_result=grab_roller_result,
        output_dir=output_dir,
        progress_centers=progress_centers,
    )
    print("[Done]")
    print(f"  output_dir: {output_dir}")
    print(f"  numerical_results: {result_path}")
    print(f"  grab_roller_included: {grab_roller_result is not None}")


if __name__ == "__main__":
    analyze(parse_args())
