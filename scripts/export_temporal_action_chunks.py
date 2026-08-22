#!/usr/bin/env python3
"""Export per-frame sampled action chunks on expert episodes.

This script is intentionally separate from eval_temporal_action_curves.py.  It
does not make plots.  It stores the actual model-sampled final action chunks and
the matching expert action chunks so downstream analysis can be done offline
without re-running the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tqdm_loggable.auto as tqdm

import eval_dataset_loss_common as common
import eval_temporal_action_curves as temporal
from openpi.training import config as _config
from openpi.training import sharding


ACTION_DIM_LABELS = (
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_gripper",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, mode="full")
    parser.set_defaults(
        task_name="place_fan",
        task_config="demo_clean",
        data_scope="source",
        support_view="ego",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=50,
        help="Use only the first N selected episodes. This exporter requires exactly one task.",
    )
    parser.add_argument(
        "--support-mode",
        choices=["enabled", "masked"],
        default="enabled",
        help="Only used with --model-kind support_caption.",
    )
    parser.add_argument(
        "--support-selection",
        choices=["random", "first", "round"],
        default="random",
        help="Only used with --model-kind support_caption.",
    )
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument("--support-round-id", type=int, default=0)
    parser.add_argument(
        "--chunk-progress-mode",
        choices=["eval_step_limit", "expert"],
        default="eval_step_limit",
        help="Progress value fed to support-caption models; pi05 does not consume it.",
    )
    parser.add_argument(
        "--step-limit-file",
        default=None,
        help="Path to RoboTwin configs/_eval_step_limit.yml. If missing, expert progress is used.",
    )
    parser.add_argument(
        "--no-flow-loss",
        action="store_true",
        help="Skip flow/vector-field loss and only save sampled final action chunks.",
    )
    parser.add_argument(
        "--npz-name",
        default="chunks.npz",
        help="Filename inside --output-dir for the saved arrays.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite an existing chunks npz and sidecar files in --output-dir.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test limit after episode selection. Omit for full export.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pad_selection_to_full_batches(
    selection: common.EvalSelection,
    batch_size: int,
) -> tuple[common.EvalSelection, int]:
    remainder = len(selection.base_indices) % batch_size
    if remainder == 0:
        return selection, 0
    pad_count = batch_size - remainder

    def pad_array(array: np.ndarray) -> np.ndarray:
        return np.concatenate([array, np.repeat(array[-1:], pad_count, axis=0)])

    train_positions = (
        None
        if selection.train_positions is None
        else pad_array(np.asarray(selection.train_positions, dtype=np.int64))
    )
    return (
        common.EvalSelection(
            base_indices=pad_array(selection.base_indices),
            episode_indices=pad_array(selection.episode_indices),
            frame_indices=pad_array(selection.frame_indices),
            support_round_ids=pad_array(selection.support_round_ids),
            train_positions=train_positions,
        ),
        pad_count,
    )


def _support_arrays(
    args: argparse.Namespace,
    support_records: list[dict[str, Any]],
    selection: common.EvalSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = common.manifest_lookup(support_records)
    support_ids: list[str] = []
    support_views: list[str] = []
    support_valid: list[bool] = []
    for episode_value, round_value in zip(
        selection.episode_indices,
        selection.support_round_ids,
        strict=True,
    ):
        episode_id = int(episode_value)
        round_id = int(round_value)
        if args.model_kind == "support_caption":
            record = lookup.get((episode_id, round_id), {})
            valid = (
                bool(record.get("has_support", True))
                and bool(str(record.get("support_frames_npy", "")).strip())
                and record.get("support_type") == "human"
            )
            if valid:
                support_id, support_view = common.support_group_info(
                    lookup,
                    episode_id,
                    round_id,
                    support_view_override=args.support_view,
                )
            else:
                support_id = str(record.get("support_id", "null"))
                support_view = "none"
        else:
            support_id, support_view, valid = "none", "none", False
        support_ids.append(support_id)
        support_views.append(support_view)
        support_valid.append(valid)
    return (
        np.asarray(support_ids, dtype="<U128"),
        np.asarray(support_views, dtype="<U16"),
        np.asarray(support_valid, dtype=np.bool_),
    )


def _build_effective_manifest(
    args: argparse.Namespace,
    raw_records: list[dict[str, Any]],
    selected_episode_ids: tuple[int, ...],
    episode_info: dict[int, common.EpisodeInfo],
    output_dir: Path,
) -> tuple[Path | None, list[dict[str, Any]]]:
    if args.model_kind != "support_caption":
        return None, []
    cfg_probe = _config.get_config(args.config_name)
    num_support_frames = int(getattr(cfg_probe.model, "num_support_frames", 8))
    return temporal.build_episode_fixed_manifest(
        raw_records,
        selected_episode_ids,
        episode_info,
        output_dir / "action_chunk_export_manifest.jsonl",
        support_mode=args.support_mode,
        support_selection=args.support_selection,
        support_seed=args.support_seed,
        support_round_id=args.support_round_id,
        support_view_override=args.support_view,
        num_support_frames=num_support_frames,
    )


def export_action_chunks(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    npz_path = output_dir / args.npz_name
    if npz_path.exists() and not args.overwrite_output:
        raise FileExistsError(f"Output already exists, pass --overwrite-output to replace: {npz_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_info, episode_order = common.load_episode_origin(args.episode_origin)
    raw_records = common.load_manifest_records(args.raw_support_manifest)
    task_names = common.parse_task_names(args.task_name)
    if task_names is None or len(task_names) != 1:
        raise ValueError(f"This exporter requires exactly one explicit --task-name, got: {args.task_name!r}")
    selected_episode_ids = common.select_episode_ids(
        episode_info,
        episode_order,
        raw_records,
        task_names=task_names,
        task_config=args.task_config,
        data_scope=args.data_scope,
    )
    selected_episode_ids = temporal.select_first_episodes_per_task(
        episode_info,
        selected_episode_ids,
        args.max_episodes_per_task,
    )
    print(
        f"[Episodes] selected={len(selected_episode_ids)} "
        f"counts={temporal._episode_task_counts(episode_info, selected_episode_ids)}"  # noqa: SLF001
    )
    print(f"[Task] explicit task={task_names[0]} config={args.task_config} scope={args.data_scope}")

    effective_manifest, support_records = _build_effective_manifest(
        args,
        raw_records,
        selected_episode_ids,
        episode_info,
        output_dir,
    )
    cfg, data_config, base_dataset, norm_stats_path = common.prepare_config_and_data(
        args,
        manifest_path=effective_manifest,
        support_rounds_per_cycle=1,
    )

    selection = common.build_full_selection(base_dataset, selected_episode_ids)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        selection = selection.subset(np.arange(min(args.max_frames, len(selection.base_indices))))

    selection_context = {
        "version": 1,
        "mode": "action_chunk_export",
        "repo_id": args.repo_id,
        "episode_origin": str(Path(args.episode_origin).expanduser().resolve()),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "data_scope": args.data_scope,
        "max_episodes_per_task": args.max_episodes_per_task,
        "selected_episode_count": len(selected_episode_ids),
        "selected_episode_ids_sha256": common._integer_sequence_sha256(selected_episode_ids),  # noqa: SLF001
        "max_frames": args.max_frames,
    }
    selection_digest = common.selection_sha256(selection, selection_context)
    common.validate_selection(selection, base_dataset, selected_episode_ids)

    padded_selection, padded_count = _pad_selection_to_full_batches(selection, args.batch_size)
    print(
        f"[Selection] requested={len(selection.base_indices)} evaluated={len(selection.base_indices)} "
        f"batch_size={args.batch_size} padded={padded_count}"
    )

    step_limits = temporal.load_step_limits(args.step_limit_file)
    expert_progress, eval_progress, model_progress = temporal.chunk_progress_values(
        selection,
        episode_info,
        mode=args.chunk_progress_mode,
        step_limits=step_limits,
    )
    padded_expert_progress, padded_eval_progress, padded_model_progress = temporal.chunk_progress_values(
        padded_selection,
        episode_info,
        mode=args.chunk_progress_mode,
        step_limits=step_limits,
    )
    del padded_expert_progress, padded_eval_progress
    if args.chunk_progress_mode == "eval_step_limit" and not step_limits:
        print("[StepLimit] no usable step-limit map; model chunk_progress equals expert progress")

    model_def, model_state, mesh, data_sharding = common.load_model_for_eval(cfg, args.step)
    eval_step = temporal.make_temporal_eval_step(
        model_def,
        compute_flow_loss=not args.no_flow_loss,
        sample_action_num_steps=args.sample_action_num_steps,
        train_preprocess=args.preprocess_mode == "train",
    )
    loader = common.create_eval_loader(
        data_config,
        base_dataset,
        padded_selection,
        data_sharding=data_sharding,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=cfg.seed,
    )
    output_transform = temporal.make_output_transform(data_config)
    rng_steps = tuple(int(value) for value in args.rng_steps.split(",") if value.strip())
    if not rng_steps:
        raise ValueError("--rng-steps is empty")

    sample_count = len(selection.base_indices)
    action_horizon = int(cfg.model.action_horizon)
    final_action_dim = len(ACTION_DIM_LABELS)
    rng_count = len(rng_steps)

    pred_actions = np.empty((rng_count, sample_count, action_horizon, final_action_dim), dtype=np.float32)
    target_actions = np.empty((sample_count, action_horizon, final_action_dim), dtype=np.float32)
    mse_t0 = np.empty((rng_count, sample_count), dtype=np.float64)
    mse_chunk_mean = np.empty((rng_count, sample_count), dtype=np.float64)
    flow_loss = np.full((rng_count, sample_count), np.nan, dtype=np.float64)
    support_input_valid = np.zeros(sample_count, dtype=np.bool_)

    num_batches = len(padded_selection.base_indices) // args.batch_size
    loader_iter = iter(loader)
    input_probe_digest: str | None = None
    with sharding.set_mesh(mesh):
        progress_bar = tqdm.tqdm(range(num_batches), desc="[ChunkExport]", dynamic_ncols=True)
        for batch_index in progress_bar:
            padded_start = batch_index * args.batch_size
            padded_end = padded_start + args.batch_size
            real_start = padded_start
            real_end = min(padded_end, sample_count)
            real_count = max(real_end - real_start, 0)
            observation, actions = next(loader_iter)
            observation = temporal._replace_chunk_progress(  # noqa: SLF001
                observation,
                padded_model_progress[padded_start:padded_end],
            )
            support_mask = getattr(observation, "support_image_mask", None)
            if real_count and support_mask is not None:
                mask_array = np.asarray(jax.device_get(support_mask))
                if mask_array.ndim == 1:
                    support_input_valid[real_start:real_end] = mask_array[:real_count].astype(bool)
                elif mask_array.ndim >= 2:
                    support_input_valid[real_start:real_end] = np.any(
                        mask_array[:real_count].astype(bool),
                        axis=tuple(range(1, mask_array.ndim)),
                    )
            if input_probe_digest is None:
                input_probe_digest = common.core_batch_sha256(observation, actions)
                print(f"[InputProbe] core_batch_sha256={input_probe_digest}")

            target_batch = temporal.load_raw_action_batch(
                base_dataset,
                padded_selection.base_indices[padded_start:padded_end],
                action_horizon=action_horizon,
                action_dim=final_action_dim,
            )
            if real_count:
                target_actions[real_start:real_end] = target_batch[:real_count]

            for rng_index, rng_step in enumerate(rng_steps):
                rng = jax.random.key(args.loss_seed)
                rng = jax.random.fold_in(rng, rng_step)
                rng = jax.random.fold_in(rng, batch_index)
                outputs = jax.tree.map(
                    np.asarray,
                    jax.device_get(eval_step(model_state, rng, observation, actions)),
                )
                pred_batch = temporal.output_to_final_actions(
                    output_transform,
                    observation,
                    outputs["sampled_actions"],
                )
                squared_error = np.square(pred_batch - target_batch)
                if real_count:
                    pred_actions[rng_index, real_start:real_end] = pred_batch[:real_count]
                    mse_t0[rng_index, real_start:real_end] = np.mean(
                        squared_error[:real_count, 0, :],
                        axis=-1,
                    )
                    mse_chunk_mean[rng_index, real_start:real_end] = np.mean(
                        squared_error[:real_count],
                        axis=(-2, -1),
                    )
                    if "flow_loss" in outputs:
                        flow_loss[rng_index, real_start:real_end] = np.asarray(
                            outputs["flow_loss"][:real_count],
                            dtype=np.float64,
                        )
            if real_count:
                progress_bar.set_postfix(
                    {
                        "mse": f"{float(np.mean(mse_chunk_mean[:, real_start:real_end])):.6f}",
                        "t0": f"{float(np.mean(mse_t0[:, real_start:real_end])):.6f}",
                    }
                )

    support_id, support_view, support_valid = _support_arrays(args, support_records, selection)
    episode_lengths = np.asarray(
        [episode_info[int(episode_id)].episode_length for episode_id in selection.episode_indices],
        dtype=np.int64,
    )
    print(
        f"[Support] manifest_valid_samples={int(np.sum(support_valid))}/{sample_count} "
        f"input_valid_samples={int(np.sum(support_input_valid))}/{sample_count}"
    )

    np.savez(
        npz_path,
        episode_index=np.asarray(selection.episode_indices, dtype=np.int64),
        frame_index=np.asarray(selection.frame_indices, dtype=np.int64),
        episode_length=episode_lengths,
        expert_progress=np.asarray(expert_progress, dtype=np.float32),
        eval_progress=np.asarray(eval_progress, dtype=np.float32),
        model_chunk_progress=np.asarray(model_progress, dtype=np.float32),
        support_round_id=np.asarray(selection.support_round_ids, dtype=np.int64),
        support_id=support_id,
        support_view=support_view,
        support_valid=support_valid,
        support_input_valid=support_input_valid,
        rng_steps=np.asarray(rng_steps, dtype=np.int64),
        pred_actions=pred_actions,
        target_actions=target_actions,
        mse_t0=mse_t0,
        mse_chunk_mean=mse_chunk_mean,
        flow_loss=flow_loss,
        action_dim_labels=np.asarray(ACTION_DIM_LABELS, dtype="<U32"),
    )

    index_rows = []
    mse_t0_mean = np.mean(mse_t0, axis=0)
    mse_chunk_mean_mean = np.mean(mse_chunk_mean, axis=0)
    flow_loss_mean = np.nanmean(flow_loss, axis=0) if not args.no_flow_loss else np.full(sample_count, np.nan)
    for index in range(sample_count):
        episode_id = int(selection.episode_indices[index])
        info = episode_info[episode_id]
        index_rows.append(
            {
                "sample_index": index,
                "task_name": info.task_name,
                "task_config": info.task_config,
                "episode_index": episode_id,
                "frame_index": int(selection.frame_indices[index]),
                "episode_length": int(info.episode_length),
                "expert_progress": float(expert_progress[index]),
                "eval_progress": float(eval_progress[index]),
                "model_chunk_progress": float(model_progress[index]),
                "support_round_id": int(selection.support_round_ids[index]),
                "support_id": str(support_id[index]),
                "support_view": str(support_view[index]),
                "support_valid": bool(support_valid[index]),
                "support_input_valid": bool(support_input_valid[index]),
                "mse_t0_mean": float(mse_t0_mean[index]),
                "mse_chunk_mean": float(mse_chunk_mean_mean[index]),
                "flow_loss_mean": float(flow_loss_mean[index]),
            }
        )
    _write_csv(output_dir / "index.csv", index_rows)

    metadata = {
        "mode": "action_chunk_export",
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
        "support_mode": args.support_mode if args.model_kind == "support_caption" else "none",
        "support_selection": args.support_selection if args.model_kind == "support_caption" else "none",
        "support_seed": args.support_seed,
        "support_round_id": args.support_round_id,
        "support_view": args.support_view if args.model_kind == "support_caption" else "none",
        "support_valid_samples": int(np.sum(support_valid)),
        "support_input_valid_samples": int(np.sum(support_input_valid)),
        "chunk_progress_mode": args.chunk_progress_mode,
        "step_limit_file": str(args.step_limit_file) if args.step_limit_file is not None else None,
        "step_limit_tasks_found": sorted(set(step_limits) & set(task_names)),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "max_episodes_per_task": args.max_episodes_per_task,
        "selected_episode_counts": temporal._episode_task_counts(episode_info, selected_episode_ids),  # noqa: SLF001
        "selection_sha256": selection_digest,
        "input_probe_sha256": input_probe_digest,
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": common.file_sha256(norm_stats_path),
        "batch_size": args.batch_size,
        "requested_samples": sample_count,
        "evaluated_samples": sample_count,
        "padded_samples": padded_count,
        "action_horizon": action_horizon,
        "model_action_dim": cfg.model.action_dim,
        "final_action_dim": final_action_dim,
        "action_space": "final_unnormalized_first_14_dims",
        "target_action_source": "raw_lerobot_action_first_14_dims",
        "pred_actions_shape": list(pred_actions.shape),
        "target_actions_shape": list(target_actions.shape),
        "sample_action_num_steps": args.sample_action_num_steps,
        "rng_steps": list(rng_steps),
        "loss_seed": args.loss_seed,
        "flow_loss": not args.no_flow_loss,
        "preprocess_mode": args.preprocess_mode,
        "npz_path": str(npz_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print("[Done]")
    print(f"  output_dir: {output_dir}")
    print(f"  npz: {npz_path}")
    print(f"  pred_actions: {pred_actions.shape}")
    print(f"  target_actions: {target_actions.shape}")


if __name__ == "__main__":
    export_action_chunks(parse_args())
