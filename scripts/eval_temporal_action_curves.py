#!/usr/bin/env python3
"""Temporal loss/action-MSE curves on expert episodes.

This evaluator is for diagnosing whether a model gets worse near the end of an
expert trajectory.  It evaluates every timestep in selected expert episodes and
stores per-timestep metrics plus paper-style curve plots.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Literal

from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tqdm_loggable.auto as tqdm  # noqa: E402

import eval_dataset_loss_common as common  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import sharding  # noqa: E402


DEFAULT_TASKS = ("place_fan", "rotate_qrcode", "move_stapler_pad")
SupportMode = Literal["enabled", "masked"]
SupportSelection = Literal["random", "first", "round"]
ChunkProgressMode = Literal["eval_step_limit", "expert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, mode="full")
    parser.set_defaults(
        task_name=",".join(DEFAULT_TASKS),
        task_config="demo_clean",
        data_scope="source",
        support_view="ego",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=50,
        help="Use only the first N selected episodes per task.",
    )
    parser.add_argument(
        "--support-mode",
        choices=["enabled", "masked"],
        default="enabled",
        help="enabled uses support video; masked passes null support video with support_image_mask=false.",
    )
    parser.add_argument(
        "--support-selection",
        choices=["random", "first", "round"],
        default="random",
        help="How to choose one fixed support record per expert episode.",
    )
    parser.add_argument(
        "--support-seed",
        type=int,
        default=0,
        help="Seed for --support-selection random. Episodes are sampled in selected episode order.",
    )
    parser.add_argument(
        "--support-round-id",
        type=int,
        default=0,
        help="Round id used when --support-selection round.",
    )
    parser.add_argument(
        "--chunk-progress-mode",
        choices=["eval_step_limit", "expert"],
        default="eval_step_limit",
        help=(
            "Progress value fed into the support model. eval_step_limit matches deploy_policy.py "
            "take_action_cnt / step_lim; expert uses frame_index / episode_length."
        ),
    )
    parser.add_argument(
        "--step-limit-file",
        default=None,
        help="Path to RoboTwin configs/_eval_step_limit.yml. If missing, expert progress is used.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of progress bins used for aggregate curve CSV/plots.",
    )
    parser.add_argument(
        "--no-flow-loss",
        action="store_true",
        help="Skip flow/vector-field loss and compute final action MSE only.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "both"],
        default="both",
        help="Curve plot file format.",
    )
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _episode_task_counts(
    episode_info: dict[int, common.EpisodeInfo],
    episode_ids: Sequence[int],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for episode_id in episode_ids:
        counts[episode_info[int(episode_id)].task_name] += 1
    return dict(sorted(counts.items()))


def select_first_episodes_per_task(
    episode_info: dict[int, common.EpisodeInfo],
    episode_ids: Sequence[int],
    max_episodes_per_task: int,
) -> tuple[int, ...]:
    if max_episodes_per_task < 1:
        raise ValueError("--max-episodes-per-task must be >= 1")
    counts: dict[str, int] = defaultdict(int)
    selected: list[int] = []
    for episode_id in episode_ids:
        task_name = episode_info[int(episode_id)].task_name
        if counts[task_name] >= max_episodes_per_task:
            continue
        selected.append(int(episode_id))
        counts[task_name] += 1
    return tuple(selected)


def _record_sort_key(record: dict[str, Any]) -> tuple[int, str, str, int]:
    return (
        common._demo_index(record),  # noqa: SLF001
        str(record.get("support_id", "")),
        str(record.get("support_view", "")),
        int(record.get("support_round_id", 0)),
    )


def _is_human_support_record(record: dict[str, Any]) -> bool:
    return (
        record.get("support_type") == "human"
        and bool(record.get("has_support", True))
        and bool(str(record.get("support_frames_npy", "")).strip())
    )


def _null_support_record(
    episode_id: int,
    info: common.EpisodeInfo,
    *,
    num_support_frames: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "global_episode_index": int(episode_id),
        "support_round_id": 0,
        "task_name": info.task_name,
        "task_config": info.task_config,
        "local_episode_index": int(episode_id),
        "source_episode_index": int(episode_id),
        "episode_length": int(info.episode_length),
        "support_type": "null",
        "has_support": False,
        "support_id": "null",
        "support_view": "none",
        "support_frames_npy": "",
        "support_frame_progress": [0.0] * int(num_support_frames),
        "video_caption": "",
        "support_skip_reason": reason,
    }


def build_episode_fixed_manifest(
    raw_records: Sequence[dict[str, Any]],
    selected_episode_ids: Sequence[int],
    episode_info: dict[int, common.EpisodeInfo],
    output_path: Path,
    *,
    support_mode: SupportMode,
    support_selection: SupportSelection,
    support_seed: int,
    support_round_id: int,
    support_view_override: str,
    num_support_frames: int,
) -> tuple[Path, list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        episode_id = int(record["global_episode_index"])
        by_episode[episode_id].append(record)

    rng = np.random.default_rng(int(support_seed))
    chosen_records: list[dict[str, Any]] = []
    missing_support_count = 0
    for episode_id in selected_episode_ids:
        info = episode_info[int(episode_id)]
        if support_mode == "masked":
            chosen = _null_support_record(
                int(episode_id),
                info,
                num_support_frames=num_support_frames,
                reason="support_mode=masked",
            )
            chosen_records.append(chosen)
            continue

        candidates = [
            record for record in by_episode.get(int(episode_id), ()) if _is_human_support_record(record)
        ]
        if support_view_override != "none":
            candidates = [
                record
                for record in candidates
                if str(record.get("support_view", "")).strip().lower() == support_view_override
            ]
        candidates = sorted(
            candidates,
            key=_record_sort_key,
        )
        if not candidates:
            missing_support_count += 1
            chosen_records.append(
                _null_support_record(
                    int(episode_id),
                    info,
                    num_support_frames=num_support_frames,
                    reason="no valid human support record",
                )
            )
            continue

        if support_selection == "first":
            chosen = dict(candidates[0])
        elif support_selection == "random":
            chosen = dict(candidates[int(rng.integers(0, len(candidates)))])
        else:
            matches = [record for record in candidates if int(record.get("support_round_id", 0)) == support_round_id]
            if matches:
                chosen = dict(matches[0])
            else:
                missing_support_count += 1
                chosen = _null_support_record(
                    int(episode_id),
                    info,
                    num_support_frames=num_support_frames,
                    reason=f"support_round_id={support_round_id} missing",
                )
        chosen["support_round_id"] = 0
        chosen_records.append(chosen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in chosen_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    if missing_support_count:
        print(f"[Support] wrote null support for {missing_support_count} episodes")
    return output_path, chosen_records


def load_step_limits(path: str | Path | None) -> dict[str, int]:
    if path is None:
        default_path = Path(__file__).resolve().parents[3] / "configs" / "_eval_step_limit.yml"
        path = default_path
    path = Path(path)
    if not path.is_file():
        print(f"[StepLimit] missing {path}; falling back to expert progress")
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:  # noqa: BLE001
        print(f"[StepLimit] failed to read {path}: {type(error).__name__}: {error}; falling back to expert progress")
        return {}
    return {str(key): int(value) for key, value in data.items()}


def chunk_progress_values(
    selection: common.EvalSelection,
    episode_info: dict[int, common.EpisodeInfo],
    *,
    mode: ChunkProgressMode,
    step_limits: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_progress = np.empty(len(selection.frame_indices), dtype=np.float32)
    eval_progress = np.empty(len(selection.frame_indices), dtype=np.float32)
    model_progress = np.empty(len(selection.frame_indices), dtype=np.float32)
    for index, (episode_value, frame_value) in enumerate(
        zip(selection.episode_indices, selection.frame_indices, strict=True)
    ):
        info = episode_info[int(episode_value)]
        frame = int(frame_value)
        expert = frame / max(info.episode_length - 1, 1)
        step_limit = step_limits.get(info.task_name)
        if step_limit is None or step_limit <= 1:
            eval_value = expert
        else:
            eval_value = frame / max(step_limit - 1, 1)
        expert_progress[index] = float(np.clip(expert, 0.0, 1.0))
        eval_progress[index] = float(np.clip(eval_value, 0.0, 1.0))
        model_progress[index] = eval_progress[index] if mode == "eval_step_limit" else expert_progress[index]
    return expert_progress, eval_progress, model_progress


def make_temporal_eval_step(
    model_def,
    *,
    compute_flow_loss: bool,
    sample_action_num_steps: int,
    train_preprocess: bool,
):
    @jax.jit
    def eval_step(model_state, rng, observation: _model.Observation, actions: _model.Actions):
        model = nnx.merge(model_def, model_state)
        model.eval()
        preprocess_rng, noise_rng, time_rng, sample_rng = jax.random.split(rng, 4)
        result = {}
        if compute_flow_loss:
            processed = _model.preprocess_observation(preprocess_rng, observation, train=train_preprocess)
            robot_tokens = model._encode_robot_images(processed)  # noqa: SLF001
            support_tokens = model._encode_support_images(processed) if model.use_support_context else None  # noqa: SLF001
            caption_semantic_tokens = None
            caption_semantic_mask = None
            if model.use_caption_supervision:
                caption_semantic_tokens, caption_semantic_mask = model.compute_caption_semantic_tokens(
                    processed,
                    support_image_tokens=support_tokens,
                    robot_image_tokens=robot_tokens,
                )
            action_loss = model._compute_action_loss(  # noqa: SLF001
                noise_rng,
                time_rng,
                processed,
                actions,
                support_tokens,
                robot_tokens,
                caption_semantic_tokens,
                caption_semantic_mask,
            )
            result["flow_loss"] = jnp.mean(action_loss, axis=-1)
        result["sampled_actions"] = model.sample_actions(
            sample_rng,
            observation,
            num_steps=sample_action_num_steps,
        )
        return result

    return eval_step


def make_output_transform(data_config: _config.DataConfig):
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ]
    )


def output_to_final_actions(output_transform, observation: _model.Observation, sampled_actions: np.ndarray) -> np.ndarray:
    # The policy runtime removes the batch dimension before applying output
    # transforms.  In particular, AlohaOutputs expects actions shaped
    # [action_horizon, action_dim] and slices ``actions[:, :14]``.  Applying the
    # same transform to a batched [B, action_horizon, action_dim] array would
    # slice the time axis instead, so keep this path sample-wise.
    states = np.asarray(jax.device_get(observation.state))
    sampled_actions = np.asarray(sampled_actions)
    final_actions = []
    for state, actions in zip(states, sampled_actions, strict=True):
        transformed = output_transform(
            {
                "state": state,
                "actions": actions,
            }
        )
        final_actions.append(np.asarray(transformed["actions"], dtype=np.float32))
    return np.stack(final_actions, axis=0)


def load_raw_action_batch(
    base_dataset,
    base_indices: Sequence[int],
    *,
    action_horizon: int,
    action_dim: int = 14,
) -> np.ndarray:
    actions = np.empty((len(base_indices), action_horizon, action_dim), dtype=np.float32)
    for output_index, base_index in enumerate(base_indices):
        sample = base_dataset[int(base_index)]
        if "action" not in sample:
            raise KeyError(f"Raw dataset sample {base_index} has no 'action' field")
        action = _as_numpy(sample["action"]).astype(np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[0] < action_horizon:
            pad = np.repeat(action[-1:, :], action_horizon - action.shape[0], axis=0)
            action = np.concatenate([action, pad], axis=0)
        action = action[:action_horizon]
        if action.shape[-1] < action_dim:
            pad = np.zeros((*action.shape[:-1], action_dim - action.shape[-1]), dtype=np.float32)
            action = np.concatenate([action, pad], axis=-1)
        actions[output_index] = action[:, :action_dim]
    return actions


def _replace_chunk_progress(observation: _model.Observation, values: np.ndarray) -> _model.Observation:
    progress = jnp.asarray(values[:, None], dtype=jnp.float32)
    if hasattr(observation, "replace"):
        return observation.replace(chunk_progress=progress)
    return dataclasses.replace(observation, chunk_progress=progress)


def _metric_stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "count": 0}
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "sem": std / math.sqrt(len(array)) if len(array) > 1 else 0.0,
        "count": int(len(array)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_binned_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    if bins < 1:
        raise ValueError("--bins must be >= 1")
    result: list[dict[str, Any]] = []
    for task_name in sorted({str(row["task_name"]) for row in rows}):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        for bin_index in range(bins):
            start = bin_index / bins
            end = (bin_index + 1) / bins
            if bin_index == bins - 1:
                selected = [
                    row
                    for row in task_rows
                    if start <= float(row["expert_progress"]) <= end
                ]
            else:
                selected = [
                    row
                    for row in task_rows
                    if start <= float(row["expert_progress"]) < end
                ]
            metric_values = {
                metric: _metric_stats([float(row[metric]) for row in selected])
                for metric in ("mse_t0", "mse_chunk_mean", "flow_loss")
            }
            output: dict[str, Any] = {
                "task_name": task_name,
                "bin_index": bin_index,
                "progress_start": start,
                "progress_end": end,
                "progress_center": (start + end) / 2.0,
                "sample_count": len(selected),
            }
            for metric, stats in metric_values.items():
                output[f"{metric}_mean"] = stats["mean"]
                output[f"{metric}_std"] = stats["std"]
                output[f"{metric}_sem"] = stats["sem"]
                output[f"{metric}_count"] = stats["count"]
            result.append(output)
    return result


def build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "tasks": {}}
    for group_name, group_rows in [
        ("overall", list(rows)),
        *[
            (task_name, [row for row in rows if row["task_name"] == task_name])
            for task_name in sorted({str(row["task_name"]) for row in rows})
        ],
    ]:
        group_summary: dict[str, Any] = {"sample_count": len(group_rows)}
        for metric in ("mse_t0", "mse_chunk_mean", "flow_loss"):
            stats = _metric_stats([float(row[metric]) for row in group_rows])
            group_summary[metric] = stats["mean"]
            group_summary[f"{metric}_std"] = stats["std"]
            group_summary[f"{metric}_count"] = stats["count"]
        if group_name == "overall":
            summary["overall"] = group_summary
        else:
            summary["tasks"][group_name] = group_summary
    return summary


def _plot_formats(plot_format: str) -> tuple[str, ...]:
    if plot_format == "both":
        return ("pdf", "png")
    return (plot_format,)


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def _task_color(task_name: str) -> str:
    palette = {
        "place_fan": "#0072B2",
        "rotate_qrcode": "#D55E00",
        "move_stapler_pad": "#009E73",
    }
    return palette.get(task_name, "#333333")


def save_metric_plot(
    binned_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    metric: str,
    ylabel: str,
    plot_format: str,
) -> None:
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for task_name in sorted({str(row["task_name"]) for row in binned_rows}):
        task_rows = [row for row in binned_rows if row["task_name"] == task_name and int(row["sample_count"]) > 0]
        x = np.asarray([float(row["progress_center"]) for row in task_rows])
        y = np.asarray([float(row[f"{metric}_mean"]) for row in task_rows])
        sem = np.asarray([float(row[f"{metric}_sem"]) for row in task_rows])
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        x, y, sem = x[finite], y[finite], sem[finite]
        color = _task_color(task_name)
        ax.plot(x, y, label=task_name, color=color, linewidth=2.0)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0.0)
    ax.set_xlabel("Expert trajectory progress")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for suffix in _plot_formats(plot_format):
        fig.savefig(plot_dir / f"temporal_{metric}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def save_task_metric_plots(
    binned_rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    plot_format: str,
) -> None:
    metric_specs = (
        ("mse_chunk_mean", "Final action MSE (chunk mean)"),
        ("mse_t0", "Final action MSE (t0)"),
        ("flow_loss", "Flow loss"),
    )
    _setup_plot_style()
    for task_name in sorted({str(row["task_name"]) for row in binned_rows}):
        task_rows = [row for row in binned_rows if row["task_name"] == task_name and int(row["sample_count"]) > 0]
        if not task_rows:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(5.2, 6.8), sharex=True)
        color = _task_color(task_name)
        for ax, (metric, ylabel) in zip(axes, metric_specs, strict=True):
            x = np.asarray([float(row["progress_center"]) for row in task_rows])
            y = np.asarray([float(row[f"{metric}_mean"]) for row in task_rows])
            sem = np.asarray([float(row[f"{metric}_sem"]) for row in task_rows])
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.plot(x[finite], y[finite], color=color, linewidth=2.0)
                ax.fill_between(
                    x[finite],
                    y[finite] - sem[finite],
                    y[finite] + sem[finite],
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                )
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, 1.0)
        axes[0].set_title(task_name)
        axes[-1].set_xlabel("Expert trajectory progress")
        fig.tight_layout()
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for suffix in _plot_formats(plot_format):
            fig.savefig(plot_dir / f"{task_name}_temporal_metrics.{suffix}", bbox_inches="tight")
        plt.close(fig)


def save_plots(binned_rows: Sequence[dict[str, Any]], output_dir: Path, *, plot_format: str) -> None:
    save_metric_plot(
        binned_rows,
        output_dir,
        metric="mse_chunk_mean",
        ylabel="Final action MSE (chunk mean)",
        plot_format=plot_format,
    )
    save_metric_plot(
        binned_rows,
        output_dir,
        metric="mse_t0",
        ylabel="Final action MSE (t0)",
        plot_format=plot_format,
    )
    save_metric_plot(
        binned_rows,
        output_dir,
        metric="flow_loss",
        ylabel="Flow loss",
        plot_format=plot_format,
    )
    save_task_metric_plots(binned_rows, output_dir, plot_format=plot_format)


def evaluate_temporal_curves(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_info, episode_order = common.load_episode_origin(args.episode_origin)
    raw_records = common.load_manifest_records(args.raw_support_manifest)
    task_names = common.parse_task_names(args.task_name)
    if task_names is None:
        task_names = DEFAULT_TASKS
        args.task_name = ",".join(task_names)
    selected_episode_ids = common.select_episode_ids(
        episode_info,
        episode_order,
        raw_records,
        task_names=task_names,
        task_config=args.task_config,
        data_scope=args.data_scope,
    )
    selected_episode_ids = select_first_episodes_per_task(
        episode_info,
        selected_episode_ids,
        args.max_episodes_per_task,
    )
    print(f"[Episodes] selected={len(selected_episode_ids)} counts={_episode_task_counts(episode_info, selected_episode_ids)}")

    support_records: list[dict[str, Any]]
    effective_manifest: Path | None
    if args.model_kind == "support_caption":
        cfg_probe = _config.get_config(args.config_name)
        num_support_frames = int(getattr(cfg_probe.model, "num_support_frames", 8))
        effective_manifest, support_records = build_episode_fixed_manifest(
            raw_records,
            selected_episode_ids,
            episode_info,
            output_dir / "temporal_eval_manifest.jsonl",
            support_mode=args.support_mode,
            support_selection=args.support_selection,
            support_seed=args.support_seed,
            support_round_id=args.support_round_id,
            support_view_override=args.support_view,
            num_support_frames=num_support_frames,
        )
    else:
        effective_manifest = None
        support_records = []

    cfg, data_config, base_dataset, norm_stats_path = common.prepare_config_and_data(
        args,
        manifest_path=effective_manifest,
        support_rounds_per_cycle=1,
    )
    selection = common.build_full_selection(base_dataset, selected_episode_ids)
    selection_context = {
        "version": 1,
        "mode": "temporal",
        "repo_id": args.repo_id,
        "episode_origin": str(Path(args.episode_origin).expanduser().resolve()),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "data_scope": args.data_scope,
        "max_episodes_per_task": args.max_episodes_per_task,
        "selected_episode_count": len(selected_episode_ids),
        "selected_episode_ids_sha256": common._integer_sequence_sha256(selected_episode_ids),  # noqa: SLF001
    }
    selection_digest = common.selection_sha256(selection, selection_context)
    common.validate_selection(selection, base_dataset, selected_episode_ids)

    requested_count = len(selection.base_indices)
    selection = selection.truncate_full_batches(args.batch_size, args.max_batches)
    dropped = requested_count - len(selection.base_indices)
    print(
        f"[Selection] requested={requested_count} evaluated={len(selection.base_indices)} "
        f"batch_size={args.batch_size} dropped_or_limited={dropped}"
    )

    step_limits = load_step_limits(args.step_limit_file)
    expert_progress, eval_progress, model_progress = chunk_progress_values(
        selection,
        episode_info,
        mode=args.chunk_progress_mode,
        step_limits=step_limits,
    )
    if args.chunk_progress_mode == "eval_step_limit" and not step_limits:
        print("[StepLimit] no usable step-limit map; model chunk_progress equals expert progress")

    model_def, model_state, mesh, data_sharding = common.load_model_for_eval(cfg, args.step)
    eval_step = make_temporal_eval_step(
        model_def,
        compute_flow_loss=not args.no_flow_loss,
        sample_action_num_steps=args.sample_action_num_steps,
        train_preprocess=args.preprocess_mode == "train",
    )
    loader = common.create_eval_loader(
        data_config,
        base_dataset,
        selection,
        data_sharding=data_sharding,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=cfg.seed,
    )
    output_transform = make_output_transform(data_config)
    lookup = common.manifest_lookup(support_records)
    rng_steps = tuple(int(value) for value in args.rng_steps.split(",") if value.strip())
    if not rng_steps:
        raise ValueError("--rng-steps is empty")

    rows: list[dict[str, Any]] = []
    num_batches = len(selection.base_indices) // args.batch_size
    loader_iter = iter(loader)
    input_probe_digest: str | None = None
    with sharding.set_mesh(mesh):
        progress_bar = tqdm.tqdm(range(num_batches), desc="[TemporalEval]", dynamic_ncols=True)
        for batch_index in progress_bar:
            start = batch_index * args.batch_size
            end = start + args.batch_size
            observation, actions = next(loader_iter)
            observation = _replace_chunk_progress(observation, model_progress[start:end])
            if input_probe_digest is None:
                input_probe_digest = common.core_batch_sha256(observation, actions)
                print(f"[InputProbe] core_batch_sha256={input_probe_digest}")

            target_actions = load_raw_action_batch(
                base_dataset,
                selection.base_indices[start:end],
                action_horizon=cfg.model.action_horizon,
                action_dim=14,
            )
            mse_t0_by_rng = []
            mse_chunk_by_rng = []
            flow_by_rng = []
            for rng_step in rng_steps:
                rng = jax.random.key(args.loss_seed)
                rng = jax.random.fold_in(rng, rng_step)
                rng = jax.random.fold_in(rng, batch_index)
                outputs = jax.tree.map(
                    np.asarray,
                    jax.device_get(eval_step(model_state, rng, observation, actions)),
                )
                pred_actions = output_to_final_actions(
                    output_transform,
                    observation,
                    outputs["sampled_actions"],
                )
                squared_error = np.square(pred_actions - target_actions)
                mse_t0_by_rng.append(np.mean(squared_error[:, 0, :], axis=-1))
                mse_chunk_by_rng.append(np.mean(squared_error, axis=(-2, -1)))
                if "flow_loss" in outputs:
                    flow_by_rng.append(np.asarray(outputs["flow_loss"], dtype=np.float64))
            mse_t0 = np.mean(np.stack(mse_t0_by_rng, axis=0), axis=0)
            mse_chunk = np.mean(np.stack(mse_chunk_by_rng, axis=0), axis=0)
            flow_loss = (
                np.mean(np.stack(flow_by_rng, axis=0), axis=0)
                if flow_by_rng
                else np.full((args.batch_size,), np.nan, dtype=np.float64)
            )

            for local_index, (episode_value, frame_value, round_value) in enumerate(
                zip(
                    selection.episode_indices[start:end],
                    selection.frame_indices[start:end],
                    selection.support_round_ids[start:end],
                    strict=True,
                )
            ):
                episode_id = int(episode_value)
                frame_index = int(frame_value)
                info = episode_info[episode_id]
                support_record = lookup.get((episode_id, int(round_value)), {})
                if args.model_kind == "support_caption":
                    support_id, support_view = common.support_group_info(
                        lookup,
                        episode_id,
                        int(round_value),
                        support_view_override=args.support_view,
                    )
                    support_valid = bool(support_record.get("support_type") == "human")
                else:
                    support_id, support_view, support_valid = "none", "none", False
                rows.append(
                    {
                        "task_name": info.task_name,
                        "task_config": info.task_config,
                        "episode_index": episode_id,
                        "frame_index": frame_index,
                        "episode_length": int(info.episode_length),
                        "expert_progress": _safe_float(expert_progress[start + local_index]),
                        "eval_progress": _safe_float(eval_progress[start + local_index]),
                        "model_chunk_progress": _safe_float(model_progress[start + local_index]),
                        "support_round_id": int(round_value),
                        "support_id": support_id,
                        "support_view": support_view,
                        "support_valid": support_valid,
                        "mse_t0": _safe_float(mse_t0[local_index]),
                        "mse_chunk_mean": _safe_float(mse_chunk[local_index]),
                        "flow_loss": _safe_float(flow_loss[local_index]),
                    }
                )
            progress_bar.set_postfix(
                {
                    "mse": f"{float(np.nanmean(mse_chunk)):.6f}",
                    "t0": f"{float(np.nanmean(mse_t0)):.6f}",
                }
            )

    binned_rows = build_binned_rows(rows, bins=args.bins)
    summary = build_summary(rows)
    write_csv(output_dir / "per_timestep.csv", rows)
    write_csv(output_dir / "binned_curves.csv", binned_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    metadata = {
        "mode": "temporal",
        "model_kind": args.model_kind,
        "config_name": args.config_name,
        "exp_name": args.exp_name,
        "step": str(args.step),
        "checkpoint_base_dir": str(cfg.checkpoint_base_dir),
        "checkpoint_dir": str(cfg.checkpoint_dir / str(args.step)),
        "repo_id": args.repo_id,
        "episode_origin": str(args.episode_origin),
        "raw_support_manifest": str(args.raw_support_manifest),
        "effective_support_manifest": None if effective_manifest is None else str(effective_manifest),
        "support_mode": args.support_mode,
        "support_selection": args.support_selection,
        "support_seed": args.support_seed,
        "support_round_id": args.support_round_id,
        "support_view": args.support_view,
        "chunk_progress_mode": args.chunk_progress_mode,
        "step_limit_file": str(args.step_limit_file) if args.step_limit_file is not None else None,
        "step_limit_tasks_found": sorted(set(step_limits) & set(task_names)),
        "plot_format": args.plot_format,
        "bins": args.bins,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "max_episodes_per_task": args.max_episodes_per_task,
        "selected_episode_counts": _episode_task_counts(episode_info, selected_episode_ids),
        "selection_sha256": selection_digest,
        "input_probe_sha256": input_probe_digest,
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": common.file_sha256(norm_stats_path),
        "batch_size": args.batch_size,
        "requested_samples": requested_count,
        "evaluated_samples": len(selection.base_indices),
        "dropped_or_limited_samples": dropped,
        "action_horizon": cfg.model.action_horizon,
        "model_action_dim": cfg.model.action_dim,
        "final_action_dim": 14,
        "sample_action_num_steps": args.sample_action_num_steps,
        "rng_steps": list(rng_steps),
        "loss_seed": args.loss_seed,
        "flow_loss": not args.no_flow_loss,
        "preprocess_mode": args.preprocess_mode,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    save_plots(binned_rows, output_dir, plot_format=args.plot_format)

    print("[Done]")
    print(f"  output_dir: {output_dir}")
    print(f"  overall: {summary['overall']}")
    print(f"  plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    evaluate_temporal_curves(parse_args())
