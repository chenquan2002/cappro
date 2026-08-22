#!/usr/bin/env python3
"""Shared dataset-loss evaluation utilities for the support-caption policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm_loggable.auto as tqdm

from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding

EvalMode = Literal["uniform", "full", "trainlike"]
ModelKind = Literal["support_caption", "pi05"]


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task_name: str
    task_config: str
    episode_length: int


@dataclasses.dataclass(frozen=True)
class EvalSelection:
    base_indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    support_round_ids: np.ndarray
    train_positions: np.ndarray | None = None

    def truncate_full_batches(self, batch_size: int, max_batches: int | None) -> EvalSelection:
        num_batches = len(self.base_indices) // batch_size
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)
        size = num_batches * batch_size
        if size <= 0:
            raise ValueError(
                f"Evaluation selection has {len(self.base_indices)} samples, smaller than batch_size={batch_size}"
            )

        def take(value):
            return None if value is None else value[:size]

        return EvalSelection(
            base_indices=self.base_indices[:size],
            episode_indices=self.episode_indices[:size],
            frame_indices=self.frame_indices[:size],
            support_round_ids=self.support_round_ids[:size],
            train_positions=take(self.train_positions),
        )

    def subset(self, indices: np.ndarray) -> EvalSelection:
        indices = np.asarray(indices, dtype=np.int64)
        train_positions = None if self.train_positions is None else self.train_positions[indices]
        return EvalSelection(
            base_indices=self.base_indices[indices],
            episode_indices=self.episode_indices[indices],
            frame_indices=self.frame_indices[indices],
            support_round_ids=self.support_round_ids[indices],
            train_positions=train_positions,
        )


@dataclasses.dataclass
class RunningLossStats:
    action_sum: float = 0.0
    action_sumsq: float = 0.0
    action_count: int = 0
    sample_count: int = 0
    sampled_action_mse_sum: float = 0.0
    sampled_action_mse_sumsq: float = 0.0
    sampled_action_mse_t0_sum: float = 0.0
    sampled_action_mse_t0_sumsq: float = 0.0
    sampled_action_mse_count: int = 0
    caption_nll_sum: float = 0.0
    caption_correct_sum: float = 0.0
    caption_token_count: int = 0
    hand_nll_sum: float = 0.0
    hand_correct_sum: float = 0.0
    hand_token_count: int = 0
    caption_valid_samples: int = 0
    support_valid_samples: int = 0

    def update(
        self,
        *,
        action_loss: float | None,
        caption_loss: float,
        caption_accuracy: float,
        caption_token_count: int,
        hand_loss: float,
        hand_accuracy: float,
        hand_token_count: int,
        support_valid: bool,
        sampled_action_mse: float | None = None,
        sampled_action_mse_t0: float | None = None,
    ) -> None:
        self.sample_count += 1
        if action_loss is not None:
            self.action_sum += float(action_loss)
            self.action_sumsq += float(action_loss) ** 2
            self.action_count += 1
        if sampled_action_mse is not None:
            sampled_action_mse = float(sampled_action_mse)
            sampled_action_mse_t0 = float(sampled_action_mse_t0 if sampled_action_mse_t0 is not None else sampled_action_mse)
            self.sampled_action_mse_sum += sampled_action_mse
            self.sampled_action_mse_sumsq += sampled_action_mse**2
            self.sampled_action_mse_t0_sum += sampled_action_mse_t0
            self.sampled_action_mse_t0_sumsq += sampled_action_mse_t0**2
            self.sampled_action_mse_count += 1
        self.caption_nll_sum += float(caption_loss) * int(caption_token_count)
        self.caption_correct_sum += float(caption_accuracy) * int(caption_token_count)
        self.caption_token_count += int(caption_token_count)
        self.hand_nll_sum += float(hand_loss) * int(hand_token_count)
        self.hand_correct_sum += float(hand_accuracy) * int(hand_token_count)
        self.hand_token_count += int(hand_token_count)
        self.caption_valid_samples += int(caption_token_count > 0)
        self.support_valid_samples += int(support_valid)

    def finalize(
        self,
        caption_loss_weight: float,
        *,
        include_caption: bool = True,
    ) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "sample_count": self.sample_count,
        }
        if self.action_count:
            action_denominator = max(self.action_count, 1)
            action_mean = self.action_sum / action_denominator
            action_variance = max(self.action_sumsq / action_denominator - action_mean**2, 0.0)
            result.update(
                {
                    "action_loss": action_mean,
                    "action_loss_std": math.sqrt(action_variance),
                }
            )
        if self.sampled_action_mse_count:
            mse_denominator = self.sampled_action_mse_count
            sampled_mean = self.sampled_action_mse_sum / mse_denominator
            sampled_variance = max(self.sampled_action_mse_sumsq / mse_denominator - sampled_mean**2, 0.0)
            sampled_t0_mean = self.sampled_action_mse_t0_sum / mse_denominator
            sampled_t0_variance = max(
                self.sampled_action_mse_t0_sumsq / mse_denominator - sampled_t0_mean**2,
                0.0,
            )
            result.update(
                {
                    "sampled_action_mse": sampled_mean,
                    "sampled_action_mse_std": math.sqrt(sampled_variance),
                    "sampled_action_mse_t0": sampled_t0_mean,
                    "sampled_action_mse_t0_std": math.sqrt(sampled_t0_variance),
                    "sampled_action_mse_count": self.sampled_action_mse_count,
                }
            )
        if not include_caption:
            return result
        caption_denominator = max(self.caption_token_count, 1)
        hand_denominator = max(self.hand_token_count, 1)
        caption_loss = self.caption_nll_sum / caption_denominator
        caption_weighted_loss = caption_loss_weight * caption_loss
        result.update({
            "caption_loss": caption_loss,
            "caption_token_accuracy": self.caption_correct_sum / caption_denominator,
            "caption_hand_side_loss": self.hand_nll_sum / hand_denominator,
            "caption_hand_side_accuracy": self.hand_correct_sum / hand_denominator,
            "caption_weighted_loss": caption_weighted_loss,
            "joint_loss": result.get("action_loss", 0.0) + caption_weighted_loss,
            "caption_token_count": self.caption_token_count,
            "caption_hand_side_token_count": self.hand_token_count,
            "caption_valid_samples": self.caption_valid_samples,
            "support_valid_samples": self.support_valid_samples,
        })
        return result


class IndexedSupportDataset:
    """Select base samples and attach an explicit support round to each one."""

    def __init__(self, dataset, base_indices: np.ndarray, support_round_ids: np.ndarray):
        if len(base_indices) != len(support_round_ids):
            raise ValueError("base_indices and support_round_ids must have the same length")
        self._dataset = dataset
        self._base_indices = np.asarray(base_indices, dtype=np.int64)
        self._support_round_ids = np.asarray(support_round_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index):
        idx = int(index.__index__() if hasattr(index, "__index__") else index)
        item = dict(self._dataset[int(self._base_indices[idx])])
        item["support_round_id"] = np.asarray(self._support_round_ids[idx], dtype=np.int64)
        return item


@contextlib.contextmanager
def temporary_environment(**updates: str):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def add_common_args(parser: argparse.ArgumentParser, *, mode: EvalMode) -> None:
    parser.add_argument("config_name", help="Train config name.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument(
        "--model-kind",
        choices=["support_caption", "pi05"],
        default="support_caption",
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        default=None,
        help="Override the config checkpoint root without moving model files.",
    )
    parser.add_argument(
        "--norm-asset-id",
        default="source_data_iclpi_repo",
        help="Checkpoint asset directory containing norm_stats.json.",
    )
    parser.add_argument("--episode-origin", required=True)
    parser.add_argument("--raw-support-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-file",
        default=None,
        help="Shared sample selection used by both compared models.",
    )
    parser.add_argument(
        "--require-existing-selection",
        action="store_true",
        help="Fail instead of creating a missing shared selection.",
    )
    parser.add_argument("--repo-id", default=os.getenv("REPO_ID", "source_data_hovapi_repo"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--episode-chunk-size", type=int, default=None)
    parser.add_argument("--episode-chunk-index", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rng-steps", default="0")
    parser.add_argument("--loss-seed", type=int, default=12345)
    parser.add_argument(
        "--sample-action-mse",
        action="store_true",
        help="Also sample actions with model.sample_actions() and report MSE to expert actions.",
    )
    parser.add_argument(
        "--sample-action-mse-only",
        action="store_true",
        help="Only compute sampled action MSE; skip flow action loss and caption metrics.",
    )
    parser.add_argument(
        "--sample-action-num-steps",
        type=int,
        default=10,
        help="Denoising/integration steps used by sample_actions() when --sample-action-mse is enabled.",
    )
    parser.add_argument("--preprocess-mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--task-name", default="all", help="One task, comma-separated tasks, or all.")
    parser.add_argument(
        "--task-config",
        choices=["all", "demo_clean", "demo_randomized"],
        default="all",
    )
    parser.add_argument("--data-scope", choices=["source", "all"], default="source")
    parser.add_argument(
        "--support-view",
        choices=["none", "ego", "front", "left", "right"],
        default="none" if mode == "trainlike" else "ego",
    )
    parser.add_argument("--caption-max-len", type=int, default=int(os.getenv("CAPTION_MAX_LEN", "96")))
    parser.add_argument(
        "--caption-loss-weight",
        type=float,
        default=float(os.getenv("CAPTION_LOSS_WEIGHT", "0.1")),
    )


def parse_task_names(value: str) -> tuple[str, ...] | None:
    names = tuple(sorted({part.strip() for part in value.split(",") if part.strip()}))
    if not names or names == ("all",):
        return None
    if "all" in names:
        raise ValueError("--task-name cannot combine all with explicit tasks")
    return names


def load_episode_origin(path: str | Path) -> tuple[dict[int, EpisodeInfo], tuple[int, ...]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"episode_origin not found: {path}")
    mapping: dict[int, EpisodeInfo] = {}
    order: list[int] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = int(record["global_episode_index"])
            mapping[episode_index] = EpisodeInfo(
                episode_index=episode_index,
                task_name=str(record["task_name"]),
                task_config=str(record["task_config"]),
                episode_length=int(record["episode_length"]),
            )
            order.append(episode_index)
    if not mapping:
        raise ValueError(f"Empty episode_origin: {path}")
    return mapping, tuple(order)


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"support manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError(f"Empty support manifest: {path}")
    return records


def human_support_task_names(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(record["task_name"])
        for record in records
        if record.get("support_type") == "human" and bool(record.get("has_support", True))
    }


def select_episode_ids(
    episode_info: dict[int, EpisodeInfo],
    episode_order: Sequence[int],
    manifest_records: Sequence[dict[str, Any]],
    *,
    task_names: tuple[str, ...] | None,
    task_config: str,
    data_scope: str,
) -> tuple[int, ...]:
    source_tasks = set(getattr(_data_loader, "_SOURCE_TASKS", ()))
    supported_tasks = human_support_task_names(manifest_records)
    if task_names is None:
        allowed_tasks = source_tasks if data_scope == "source" else supported_tasks
    else:
        allowed_tasks = set(task_names)
        missing = allowed_tasks - {info.task_name for info in episode_info.values()}
        if missing:
            raise ValueError(f"Unknown task names: {sorted(missing)}")
        if data_scope == "source":
            outside_source = allowed_tasks - source_tasks
            if outside_source:
                raise ValueError(
                    f"Tasks {sorted(outside_source)} are outside source scope; use --data-scope all"
                )

    selected = tuple(
        episode_index
        for episode_index in episode_order
        if episode_info[episode_index].task_name in allowed_tasks
        and (task_config == "all" or episode_info[episode_index].task_config == task_config)
    )
    if not selected:
        raise ValueError(
            f"No episodes selected for task_name={task_names or 'all'}, task_config={task_config}, "
            f"data_scope={data_scope}"
        )
    return selected


def _demo_index(record: dict[str, Any]) -> int:
    text = str(record.get("support_id", ""))
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else 10**9


def build_first_demo_manifest(
    manifest_records: Sequence[dict[str, Any]],
    selected_episode_ids: Sequence[int],
    output_path: str | Path,
) -> tuple[Path, dict[int, dict[str, Any]]]:
    selected_set = set(selected_episode_ids)
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        episode_index = int(record["global_episode_index"])
        if episode_index in selected_set:
            by_episode[episode_index].append(record)

    chosen_by_episode: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for episode_index in selected_episode_ids:
        records = by_episode.get(episode_index, [])
        human = [
            record
            for record in records
            if record.get("support_type") == "human"
            and bool(record.get("has_support", True))
            and str(record.get("support_frames_npy", "")).strip()
        ]
        candidates = human or records
        if not candidates:
            missing.append(episode_index)
            continue
        chosen = dict(
            min(
                candidates,
                key=lambda record: (
                    _demo_index(record),
                    str(record.get("support_id", "")),
                    int(record.get("support_round_id", 0)),
                ),
            )
        )
        chosen["support_round_id"] = 0
        chosen_by_episode[episode_index] = chosen

    if missing:
        raise ValueError(f"Manifest has no records for {len(missing)} selected episodes: {missing[:10]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for episode_index in selected_episode_ids:
            file.write(json.dumps(chosen_by_episode[episode_index], ensure_ascii=False) + "\n")
    return output_path, chosen_by_episode


def _unwrap_lerobot_dataset(dataset):
    if hasattr(dataset, "episode_data_index"):
        return dataset
    for attribute in ("_dataset", "dataset", "base_dataset"):
        inner = getattr(dataset, attribute, None)
        if inner is not None:
            found = _unwrap_lerobot_dataset(inner)
            if found is not None:
                return found
    return None


def episode_spans(dataset, episode_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    lerobot_dataset = _unwrap_lerobot_dataset(dataset)
    if lerobot_dataset is None:
        raise TypeError("Cannot find LeRobot dataset with episode_data_index")
    episode_from = lerobot_dataset.episode_data_index["from"]
    episode_to = lerobot_dataset.episode_data_index["to"]
    starts = np.asarray([int(episode_from[index]) for index in episode_ids], dtype=np.int64)
    ends = np.asarray([int(episode_to[index]) for index in episode_ids], dtype=np.int64)
    if np.any(ends <= starts):
        raise ValueError("Selected episodes contain empty frame spans")
    return starts, ends


def build_uniform_selection(
    dataset,
    episode_ids: Sequence[int],
    *,
    samples_per_episode: int,
) -> EvalSelection:
    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be >= 1")
    starts, ends = episode_spans(dataset, episode_ids)
    base_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    for episode_id, start, end in zip(episode_ids, starts, ends, strict=True):
        length = int(end - start)
        count = min(samples_per_episode, length)
        frames = np.linspace(0, length - 1, count).round().astype(np.int64)
        base_indices.append(start + frames)
        episode_indices.append(np.full(count, episode_id, dtype=np.int64))
        frame_indices.append(frames)
    base = np.concatenate(base_indices)
    return EvalSelection(
        base_indices=base,
        episode_indices=np.concatenate(episode_indices),
        frame_indices=np.concatenate(frame_indices),
        support_round_ids=np.zeros_like(base),
    )


def build_full_selection(dataset, episode_ids: Sequence[int]) -> EvalSelection:
    starts, ends = episode_spans(dataset, episode_ids)
    base_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    for episode_id, start, end in zip(episode_ids, starts, ends, strict=True):
        length = int(end - start)
        base_indices.append(np.arange(start, end, dtype=np.int64))
        episode_indices.append(np.full(length, episode_id, dtype=np.int64))
        frame_indices.append(np.arange(length, dtype=np.int64))
    base = np.concatenate(base_indices)
    return EvalSelection(
        base_indices=base,
        episode_indices=np.concatenate(episode_indices),
        frame_indices=np.concatenate(frame_indices),
        support_round_ids=np.zeros_like(base),
    )


def _training_permutation_prefix(
    expanded_len: int,
    total_seen: int,
    *,
    seed: int,
) -> np.ndarray:
    """Reproduce DataLoader shuffle, including its per-iterator base-seed draw."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    remaining = total_seen
    chunks: list[np.ndarray] = []
    while remaining > 0:
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        permutation = torch.randperm(expanded_len, generator=generator, dtype=torch.int64)
        take = min(remaining, expanded_len)
        chunks.append(permutation[:take].cpu().numpy())
        remaining -= take
        del permutation
    return np.concatenate(chunks)


def _selection_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_sha256(selection: EvalSelection, context: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in ("base_indices", "episode_indices", "frame_indices", "support_round_ids"):
        values = np.ascontiguousarray(getattr(selection, name), dtype=np.int64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    if selection.train_positions is not None:
        values = np.ascontiguousarray(selection.train_positions, dtype=np.int64)
        digest.update(b"train_positions")
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_selection(
    selection: EvalSelection,
    dataset,
    selected_episode_ids: Sequence[int],
) -> None:
    size = len(selection.base_indices)
    fields = {
        "episode_indices": selection.episode_indices,
        "frame_indices": selection.frame_indices,
        "support_round_ids": selection.support_round_ids,
    }
    if selection.train_positions is not None:
        fields["train_positions"] = selection.train_positions
    mismatched = {name: len(value) for name, value in fields.items() if len(value) != size}
    if mismatched:
        raise ValueError(f"Shared selection arrays have inconsistent lengths: base={size}, {mismatched}")
    if size == 0:
        raise ValueError("Shared selection is empty")

    allowed = set(map(int, selected_episode_ids))
    actual = set(map(int, selection.episode_indices))
    outside = actual - allowed
    if outside:
        raise ValueError(f"Shared selection contains unselected episodes: {sorted(outside)[:10]}")
    if np.any(selection.support_round_ids < 0):
        raise ValueError("Shared selection contains negative support round IDs")

    unique_episodes = tuple(sorted(actual))
    starts, ends = episode_spans(dataset, unique_episodes)
    span_by_episode = {
        episode: (int(start), int(end))
        for episode, start, end in zip(unique_episodes, starts, ends, strict=True)
    }
    expected_base = np.empty(size, dtype=np.int64)
    for index, (episode_value, frame_value) in enumerate(
        zip(selection.episode_indices, selection.frame_indices, strict=True)
    ):
        episode_index = int(episode_value)
        frame_index = int(frame_value)
        start, end = span_by_episode[episode_index]
        if frame_index < 0 or frame_index >= end - start:
            raise ValueError(
                f"Shared selection frame {frame_index} is outside episode {episode_index} length {end - start}"
            )
        expected_base[index] = start + frame_index
    if not np.array_equal(expected_base, selection.base_indices):
        mismatch = int(np.flatnonzero(expected_base != selection.base_indices)[0])
        raise ValueError(
            "Shared selection base index mismatch at position "
            f"{mismatch}: stored={selection.base_indices[mismatch]}, expected={expected_base[mismatch]}"
        )


def save_shared_selection(
    path: str | Path,
    selection: EvalSelection,
    context: dict[str, Any],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = selection_sha256(selection, context)
    train_positions = (
        np.asarray([], dtype=np.int64)
        if selection.train_positions is None
        else np.asarray(selection.train_positions, dtype=np.int64)
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(
            file,
            context_json=np.asarray(json.dumps(context, sort_keys=True, separators=(",", ":"))),
            selection_sha256=np.asarray(digest),
            base_indices=np.asarray(selection.base_indices, dtype=np.int64),
            episode_indices=np.asarray(selection.episode_indices, dtype=np.int64),
            frame_indices=np.asarray(selection.frame_indices, dtype=np.int64),
            support_round_ids=np.asarray(selection.support_round_ids, dtype=np.int64),
            train_positions=train_positions,
            has_train_positions=np.asarray(selection.train_positions is not None),
        )
    temporary_path.replace(path)
    print(f"[SharedSelection] saved {path} sha256={digest}")
    return digest


def load_shared_selection(
    path: str | Path,
    expected_context: dict[str, Any],
) -> tuple[EvalSelection, str]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        context = json.loads(str(np.asarray(stored["context_json"]).item()))
        if context != expected_context:
            raise ValueError(
                "Shared selection context mismatch. "
                f"stored={context}, expected={expected_context}"
            )
        has_train_positions = bool(np.asarray(stored["has_train_positions"]).item())
        selection = EvalSelection(
            base_indices=stored["base_indices"].astype(np.int64),
            episode_indices=stored["episode_indices"].astype(np.int64),
            frame_indices=stored["frame_indices"].astype(np.int64),
            support_round_ids=stored["support_round_ids"].astype(np.int64),
            train_positions=(
                stored["train_positions"].astype(np.int64)
                if has_train_positions
                else None
            ),
        )
        stored_digest = str(np.asarray(stored["selection_sha256"]).item())
    actual_digest = selection_sha256(selection, context)
    if actual_digest != stored_digest:
        raise ValueError(
            f"Shared selection checksum mismatch: stored={stored_digest}, actual={actual_digest}"
        )
    print(f"[SharedSelection] loaded {path} sha256={actual_digest}")
    return selection, actual_digest


def build_or_load_trainlike_selection(
    dataset,
    training_episode_ids: Sequence[int],
    selected_episode_ids: Sequence[int],
    replay_file: str | Path,
    *,
    train_steps: int,
    train_batch_size: int,
    support_rounds_per_cycle: int,
    eval_samples: int,
    replay_seed: int,
    sample_method: str,
    sample_seed: int,
    force_remake: bool,
) -> EvalSelection:
    starts, ends = episode_spans(dataset, training_episode_ids)
    lengths = ends - starts
    compact_ends = np.cumsum(lengths)
    compact_starts = np.concatenate([np.asarray([0], dtype=np.int64), compact_ends[:-1]])
    compact_len = int(compact_ends[-1])
    expanded_len = compact_len * support_rounds_per_cycle
    total_seen = train_steps * train_batch_size
    selected_set = set(selected_episode_ids)
    signature_payload = {
        "version": 2,
        "training_episode_ids": list(map(int, training_episode_ids)),
        "selected_episode_ids": list(map(int, selected_episode_ids)),
        "compact_len": compact_len,
        "train_steps": train_steps,
        "train_batch_size": train_batch_size,
        "support_rounds_per_cycle": support_rounds_per_cycle,
        "eval_samples": eval_samples,
        "replay_seed": replay_seed,
        "sample_method": sample_method,
        "sample_seed": sample_seed,
    }
    signature = _selection_signature(signature_payload)
    replay_file = Path(replay_file)
    if replay_file.is_file() and not force_remake:
        with np.load(replay_file, allow_pickle=False) as replay:
            stored_signature = str(np.asarray(replay["signature"]).item())
            if stored_signature == signature:
                return EvalSelection(
                    base_indices=replay["base_indices"].astype(np.int64),
                    episode_indices=replay["episode_indices"].astype(np.int64),
                    frame_indices=replay["frame_indices"].astype(np.int64),
                    support_round_ids=replay["support_round_ids"].astype(np.int64),
                    train_positions=replay["train_positions"].astype(np.int64),
                )
        print(f"[Replay] signature changed; regenerating {replay_file}")

    stream = _training_permutation_prefix(expanded_len, total_seen, seed=replay_seed)
    compact_indices = stream % compact_len
    support_round_ids = stream // compact_len
    episode_positions = np.searchsorted(compact_ends, compact_indices, side="right")
    training_episode_array = np.asarray(training_episode_ids, dtype=np.int64)
    episode_indices = training_episode_array[episode_positions]
    mask = np.fromiter((int(value) in selected_set for value in episode_indices), dtype=bool, count=len(episode_indices))
    candidate_positions = np.flatnonzero(mask)
    if not len(candidate_positions):
        raise ValueError("No selected-task samples occur in the requested train-like stream")
    count = min(eval_samples, len(candidate_positions))
    if sample_method == "linspace":
        chosen_offsets = np.linspace(0, len(candidate_positions) - 1, count).round().astype(np.int64)
    elif sample_method == "random":
        rng = np.random.default_rng(sample_seed)
        chosen_offsets = np.sort(rng.choice(len(candidate_positions), size=count, replace=False))
    else:
        raise ValueError(f"Unknown replay sample method: {sample_method}")
    train_positions = candidate_positions[chosen_offsets]
    chosen_compact = compact_indices[train_positions]
    chosen_episode_positions = episode_positions[train_positions]
    frame_indices = chosen_compact - compact_starts[chosen_episode_positions]
    base_indices = starts[chosen_episode_positions] + frame_indices
    selection = EvalSelection(
        base_indices=base_indices.astype(np.int64),
        episode_indices=episode_indices[train_positions].astype(np.int64),
        frame_indices=frame_indices.astype(np.int64),
        support_round_ids=support_round_ids[train_positions].astype(np.int64),
        train_positions=train_positions.astype(np.int64),
    )
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        replay_file,
        signature=np.asarray(signature),
        base_indices=selection.base_indices,
        episode_indices=selection.episode_indices,
        frame_indices=selection.frame_indices,
        support_round_ids=selection.support_round_ids,
        train_positions=selection.train_positions,
    )
    return selection


def prepare_config_and_data(
    args,
    *,
    manifest_path: str | Path | None,
    support_rounds_per_cycle: int,
):
    cfg = _config.get_config(args.config_name)
    if args.model_kind == "support_caption":
        model_config = dataclasses.replace(
            cfg.model,
            caption_max_len=args.caption_max_len,
            caption_loss_weight=args.caption_loss_weight,
        )
        if not getattr(model_config, "use_support_context", False):
            raise ValueError("support_caption evaluation requires a support-context model config")
        if not getattr(model_config, "use_caption_supervision", False):
            raise ValueError("support_caption evaluation requires caption supervision in the model config")
    else:
        model_config = cfg.model
        if getattr(model_config, "use_support_context", False):
            raise ValueError("pi05 evaluation requires a model config without support context")

    config_updates = {
        "exp_name": args.exp_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "model": model_config,
        "resume": True,
        "overwrite": False,
        "wandb_enabled": False,
        "fsdp_devices": 1,
    }
    if args.checkpoint_base_dir is not None:
        config_updates["checkpoint_base_dir"] = str(Path(args.checkpoint_base_dir).expanduser().resolve())
    cfg = dataclasses.replace(
        cfg,
        **config_updates,
    )

    with temporary_environment(TRAIN_DATA_SCOPE="source"):
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    data_updates: dict[str, Any] = {
        "repo_id": args.repo_id,
        "asset_id": args.norm_asset_id,
    }
    if args.model_kind == "support_caption":
        if manifest_path is None:
            raise ValueError("support_caption evaluation requires a support manifest")
        data_updates.update(
            {
                "use_support_context": True,
                "support_manifest_path": str(manifest_path),
                "support_rounds_per_cycle": support_rounds_per_cycle,
                "support_view_override": args.support_view,
                "use_caption_supervision": True,
                "caption_max_len": args.caption_max_len,
            }
        )
    else:
        data_updates.update(
            {
                "use_support_context": False,
                "support_manifest_path": None,
                "support_rounds_per_cycle": 1,
                "support_view_override": "none",
                "use_caption_supervision": False,
            }
        )
    data_config = dataclasses.replace(data_config, **data_updates)

    checkpoint_dir = cfg.checkpoint_dir / str(args.step)
    checkpoint_norm_dir = checkpoint_dir / "assets" / args.norm_asset_id
    norm_stats_path = checkpoint_norm_dir / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Checkpoint normalization stats not found: {norm_stats_path}")
    data_config = dataclasses.replace(data_config, norm_stats=_normalize.load(checkpoint_norm_dir))
    print(f"[NormStats] loaded from checkpoint: {checkpoint_norm_dir}")

    with temporary_environment(TRAIN_DATA_SCOPE="all"):
        base_dataset = _data_loader.create_torch_dataset(
            data_config,
            action_horizon=cfg.model.action_horizon,
            model_config=cfg.model,
        )
    return cfg, data_config, base_dataset, norm_stats_path


def load_model_for_eval(cfg: _config.TrainConfig, step: str):
    params_path = cfg.checkpoint_dir / str(step) / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Checkpoint params not found: {params_path}")
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = cfg.model.load(params)
    model.eval()
    model_def, model_state = nnx.split(model)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    return model_def, model_state, mesh, data_sharding


def make_eval_step(
    model_def,
    *,
    train_preprocess: bool,
    compute_flow_loss: bool,
    include_caption: bool,
    sample_action_mse: bool,
    sample_action_num_steps: int,
):
    @jax.jit
    def eval_step(model_state, rng, observation: _model.Observation, actions: _model.Actions):
        model = nnx.merge(model_def, model_state)
        model.eval()
        preprocess_rng, noise_rng, time_rng, sample_rng = jax.random.split(rng, 4)
        processed = None
        support_tokens = None
        caption_semantic_tokens = None
        caption_semantic_mask = None
        caption_metrics = None
        result = {}
        if compute_flow_loss or include_caption:
            processed = _model.preprocess_observation(preprocess_rng, observation, train=train_preprocess)
            robot_tokens = model._encode_robot_images(processed)  # noqa: SLF001
            support_tokens = model._encode_support_images(processed) if model.use_support_context else None  # noqa: SLF001
            if model.use_caption_supervision:
                if include_caption:
                    caption_metrics, caption_semantic_tokens, caption_semantic_mask = model.compute_caption_outputs(
                        processed,
                        support_image_tokens=support_tokens,
                        robot_image_tokens=robot_tokens,
                    )
                else:
                    caption_semantic_tokens, caption_semantic_mask = model.compute_caption_semantic_tokens(
                        processed,
                        support_image_tokens=support_tokens,
                        robot_image_tokens=robot_tokens,
                    )
        if compute_flow_loss:
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
            result["action_loss"] = jnp.mean(action_loss, axis=-1)
        if sample_action_mse:
            sampled_actions = model.sample_actions(
                sample_rng,
                observation,
                num_steps=sample_action_num_steps,
            )
            squared_error = jnp.square(sampled_actions - actions)
            result["sampled_action_mse"] = jnp.mean(squared_error, axis=(-2, -1))
            result["sampled_action_mse_t0"] = jnp.mean(squared_error[:, 0, :], axis=-1)
        if include_caption:
            assert processed is not None
            assert caption_metrics is not None
            result.update(caption_metrics)
            result["support_valid"] = jnp.any(processed.support_image_mask, axis=-1)
        return result

    return eval_step


def create_eval_loader(
    data_config: _config.DataConfig,
    base_dataset,
    selection: EvalSelection,
    *,
    data_sharding,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    dataset = IndexedSupportDataset(base_dataset, selection.base_indices, selection.support_round_ids)
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=False,
        sampler=None,
        num_batches=len(selection.base_indices) // batch_size,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)


def manifest_lookup(records: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(record["global_episode_index"]), int(record.get("support_round_id", 0))): record
        for record in records
    }


def support_group_info(
    lookup: dict[tuple[int, int], dict[str, Any]],
    episode_index: int,
    round_id: int,
    *,
    support_view_override: str,
) -> tuple[str, str]:
    record = lookup.get((episode_index, round_id), {})
    support_id = str(record.get("support_id", "null"))
    original_view = str(record.get("support_view", "none"))
    effective_view = support_view_override if support_view_override != "none" else original_view
    return support_id, effective_view


def _update_group(
    stats: dict[tuple[str, ...], RunningLossStats],
    key: tuple[str, ...],
    values: dict[str, Any],
    index: int,
    *,
    include_caption: bool,
) -> None:
    stats[key].update(
        action_loss=float(values["action_loss"][index]) if "action_loss" in values else None,
        caption_loss=float(values["caption_loss"][index]) if include_caption else 0.0,
        caption_accuracy=float(values["caption_token_accuracy"][index]) if include_caption else 0.0,
        caption_token_count=int(values["caption_token_count"][index]) if include_caption else 0,
        hand_loss=float(values["caption_hand_side_loss"][index]) if include_caption else 0.0,
        hand_accuracy=float(values["caption_hand_side_accuracy"][index]) if include_caption else 0.0,
        hand_token_count=int(values["caption_hand_side_token_count"][index]) if include_caption else 0,
        support_valid=bool(values["support_valid"][index]) if include_caption else False,
        sampled_action_mse=(
            float(values["sampled_action_mse"][index]) if "sampled_action_mse" in values else None
        ),
        sampled_action_mse_t0=(
            float(values["sampled_action_mse_t0"][index]) if "sampled_action_mse_t0" in values else None
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def core_batch_sha256(observation: _model.Observation, actions: _model.Actions) -> str:
    """Hash model inputs shared by pi05 and support-caption, excluding support-only fields."""
    digest = hashlib.sha256()

    def update(name: str, value) -> None:
        if value is None:
            digest.update(f"{name}:none".encode())
            return
        array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    for name in sorted(observation.images):
        update(f"images.{name}", observation.images[name])
    for name in sorted(observation.image_masks):
        update(f"image_masks.{name}", observation.image_masks[name])
    update("state", observation.state)
    update("tokenized_prompt", observation.tokenized_prompt)
    update("tokenized_prompt_mask", observation.tokenized_prompt_mask)
    update("token_ar_mask", observation.token_ar_mask)
    update("token_loss_mask", observation.token_loss_mask)
    update("actions", actions)
    return digest.hexdigest()


def evaluate_selection(
    *,
    args,
    mode: EvalMode,
    cfg: _config.TrainConfig,
    data_config: _config.DataConfig,
    base_dataset,
    selection: EvalSelection,
    selection_digest: str,
    norm_stats_path: Path,
    episode_info: dict[int, EpisodeInfo],
    support_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    sample_action_mse = bool(args.sample_action_mse or args.sample_action_mse_only)
    compute_flow_loss = not bool(args.sample_action_mse_only)
    include_caption = args.model_kind == "support_caption" and not bool(args.sample_action_mse_only)
    requested_count = len(selection.base_indices)
    requested_episode_count = len(set(map(int, selection.episode_indices)))
    selection = selection.truncate_full_batches(args.batch_size, args.max_batches)
    dropped = requested_count - len(selection.base_indices)
    print(
        f"[Selection] requested={requested_count} evaluated={len(selection.base_indices)} "
        f"batch_size={args.batch_size} dropped_or_limited={dropped}"
    )
    model_def, model_state, mesh, data_sharding = load_model_for_eval(cfg, args.step)
    eval_step = make_eval_step(
        model_def,
        train_preprocess=args.preprocess_mode == "train",
        compute_flow_loss=compute_flow_loss,
        include_caption=include_caption,
        sample_action_mse=sample_action_mse,
        sample_action_num_steps=args.sample_action_num_steps,
    )
    loader = create_eval_loader(
        data_config,
        base_dataset,
        selection,
        data_sharding=data_sharding,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=cfg.seed,
    )
    rng_steps = tuple(int(value) for value in args.rng_steps.split(",") if value.strip())
    if not rng_steps:
        raise ValueError("--rng-steps is empty")
    lookup = manifest_lookup(support_records)
    stats: dict[tuple[str, ...], RunningLossStats] = defaultdict(RunningLossStats)
    num_batches = len(selection.base_indices) // args.batch_size
    loader_iter = iter(loader)
    input_probe_digest: str | None = None

    with sharding.set_mesh(mesh):
        progress = tqdm.tqdm(range(num_batches), desc=f"[Eval:{mode}]", dynamic_ncols=True)
        for batch_index in progress:
            observation, actions = next(loader_iter)
            if input_probe_digest is None:
                input_probe_digest = core_batch_sha256(observation, actions)
                print(f"[InputProbe] core_batch_sha256={input_probe_digest}")
            outputs_by_rng = []
            for rng_step in rng_steps:
                rng = jax.random.key(args.loss_seed)
                rng = jax.random.fold_in(rng, rng_step)
                rng = jax.random.fold_in(rng, batch_index)
                outputs_by_rng.append(
                    jax.tree.map(
                        np.asarray,
                        jax.device_get(eval_step(model_state, rng, observation, actions)),
                    )
                )
            values = {
                key: np.mean(np.stack([output[key] for output in outputs_by_rng]), axis=0)
                for key in outputs_by_rng[0]
            }
            start = batch_index * args.batch_size
            end = start + args.batch_size
            for index, (episode_value, round_value) in enumerate(
                zip(
                    selection.episode_indices[start:end],
                    selection.support_round_ids[start:end],
                    strict=True,
                )
            ):
                episode_index = int(episode_value)
                round_id = int(round_value)
                info = episode_info[episode_index]
                keys = [
                    ("overall",),
                    ("task_config", info.task_config),
                    ("task", info.task_name, "all"),
                    ("task", info.task_name, info.task_config),
                ]
                if include_caption:
                    support_id, effective_view = support_group_info(
                        lookup,
                        episode_index,
                        round_id,
                        support_view_override=args.support_view,
                    )
                    keys.extend(
                        [
                            ("support_id", support_id),
                            ("support_view", effective_view),
                        ]
                    )
                if mode == "trainlike":
                    round_name = f"round_{round_id:02d}"
                    keys.extend(
                        [
                            ("support_round", round_name),
                            ("task_round", info.task_name, round_name),
                            ("task_config_round", info.task_name, info.task_config, round_name),
                        ]
                    )
                for key in keys:
                    _update_group(stats, key, values, index, include_caption=include_caption)
            if "action_loss" in values:
                progress.set_postfix({"action": f"{float(np.mean(values['action_loss'])):.6f}"})
            elif "sampled_action_mse" in values:
                progress.set_postfix({"mse": f"{float(np.mean(values['sampled_action_mse'])):.6f}"})

    summary = {
        "|".join(key): value.finalize(
            args.caption_loss_weight,
            include_caption=include_caption,
        )
        for key, value in stats.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def rows_for(prefix: str, names: Sequence[str]) -> list[dict[str, Any]]:
        rows = []
        for key, value in stats.items():
            if key and key[0] == prefix:
                rows.append(
                    {
                        **dict(zip(names, key[1:], strict=True)),
                        **value.finalize(
                            args.caption_loss_weight,
                            include_caption=include_caption,
                        ),
                    }
                )
        return sorted(rows, key=lambda row: tuple(str(row[name]) for name in names))

    write_csv(output_dir / "taskwise.csv", rows_for("task", ["task_name", "task_config"]))
    if include_caption:
        write_csv(output_dir / "support_id.csv", rows_for("support_id", ["support_id"]))
        write_csv(output_dir / "support_view.csv", rows_for("support_view", ["support_view"]))
    if mode == "trainlike":
        write_csv(output_dir / "roundwise.csv", rows_for("support_round", ["support_round"]))
        write_csv(
            output_dir / "task_roundwise.csv",
            rows_for("task_round", ["task_name", "support_round"]),
        )
        write_csv(
            output_dir / "task_config_roundwise.csv",
            rows_for("task_config_round", ["task_name", "task_config", "support_round"]),
        )

    metadata = {
        "mode": mode,
        "model_kind": args.model_kind,
        "config_name": args.config_name,
        "exp_name": args.exp_name,
        "step": str(args.step),
        "checkpoint_base_dir": str(cfg.checkpoint_base_dir),
        "checkpoint_dir": str(cfg.checkpoint_dir / str(args.step)),
        "repo_id": args.repo_id,
        "raw_support_manifest": str(args.raw_support_manifest),
        "effective_support_manifest": (
            str(data_config.support_manifest_path) if data_config.support_manifest_path is not None else None
        ),
        "selection_file": str(args.selection_file) if args.selection_file is not None else None,
        "selection_sha256": selection_digest,
        "input_probe_sha256": input_probe_digest,
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": file_sha256(norm_stats_path),
        "episode_origin": str(args.episode_origin),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "data_scope": args.data_scope,
        "support_view": args.support_view,
        "caption_max_len": args.caption_max_len,
        "caption_loss_weight": args.caption_loss_weight,
        "batch_size": args.batch_size,
        "action_horizon": cfg.model.action_horizon,
        "action_dim": cfg.model.action_dim,
        "rng_steps": list(rng_steps),
        "loss_seed": args.loss_seed,
        "sample_action_mse": bool(args.sample_action_mse or args.sample_action_mse_only),
        "sample_action_mse_only": bool(args.sample_action_mse_only),
        "sample_action_num_steps": args.sample_action_num_steps,
        "preprocess_mode": args.preprocess_mode,
        "requested_samples": requested_count,
        "evaluated_samples": len(selection.base_indices),
        "dropped_or_limited_samples": dropped,
        "requested_episodes": requested_episode_count,
        "evaluated_episodes": len(set(map(int, selection.episode_indices))),
    }
    if mode == "full" and args.episode_chunk_size is not None:
        metadata["episode_chunk_size"] = args.episode_chunk_size
        metadata["episode_chunk_index"] = args.episode_chunk_index
    if mode == "uniform":
        metadata["samples_per_episode"] = args.samples_per_episode
    elif mode == "trainlike":
        metadata.update(
            {
                "replay_file": str(args.replay_file),
                "train_steps": args.train_steps,
                "train_batch_size": args.train_batch_size,
                "support_rounds_per_cycle": args.support_rounds_per_cycle,
                "eval_samples": args.eval_samples,
                "replay_seed": cfg.seed if args.replay_seed is None else args.replay_seed,
                "replay_sample_method": args.replay_sample_method,
                "sample_seed": args.sample_seed,
            }
        )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[Done]")
    print(f"  output_dir: {output_dir}")
    print(f"  overall: {summary.get('overall')}")
    return summary


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _shared_selection_context(
    args,
    *,
    mode: EvalMode,
    selected_episode_ids: Sequence[int],
    dataset_length: int,
    replay_seed: int,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "version": 1,
        "mode": mode,
        "repo_id": args.repo_id,
        "episode_origin": str(Path(args.episode_origin).expanduser().resolve()),
        "task_name": args.task_name,
        "task_config": args.task_config,
        "data_scope": args.data_scope,
        "dataset_length": int(dataset_length),
        "selected_episode_count": len(selected_episode_ids),
        "selected_episode_ids_sha256": _integer_sequence_sha256(selected_episode_ids),
    }
    if getattr(args, "episode_chunk_size", None) is not None:
        context["episode_chunk_size"] = int(args.episode_chunk_size)
    if getattr(args, "episode_chunk_index", None) is not None:
        context["episode_chunk_index"] = int(args.episode_chunk_index)
    if mode == "uniform":
        context["samples_per_episode"] = args.samples_per_episode
    elif mode == "trainlike":
        context.update(
            {
                "train_steps": args.train_steps,
                "train_batch_size": args.train_batch_size,
                "support_rounds_per_cycle": args.support_rounds_per_cycle,
                "eval_samples": args.eval_samples,
                "replay_seed": replay_seed,
                "replay_sample_method": args.replay_sample_method,
                "sample_seed": args.sample_seed,
            }
        )
    return context


def run_mode(args, *, mode: EvalMode) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_info, episode_order = load_episode_origin(args.episode_origin)
    raw_records = load_manifest_records(args.raw_support_manifest)
    task_names = parse_task_names(args.task_name)
    source_tasks = set(getattr(_data_loader, "_SOURCE_TASKS", ()))
    if task_names is not None and args.data_scope == "source" and set(task_names) - source_tasks:
        print(
            f"[DataScope] selected target task {sorted(set(task_names) - source_tasks)}; "
            "using the full dataset with checkpoint source normalization"
        )
        args.data_scope = "all"
    selected_episode_ids = select_episode_ids(
        episode_info,
        episode_order,
        raw_records,
        task_names=task_names,
        task_config=args.task_config,
        data_scope=args.data_scope,
    )
    print(
        f"[Episodes] selected={len(selected_episode_ids)} tasks="
        f"{sorted({episode_info[index].task_name for index in selected_episode_ids})}"
    )

    if mode == "full" and args.episode_chunk_size is not None:
        if args.episode_chunk_size < 1:
            raise ValueError("--episode-chunk-size must be >= 1")
        if args.episode_chunk_index is None or args.episode_chunk_index < 0:
            raise ValueError("--episode-chunk-index must be a non-negative integer")
        chunk_start = args.episode_chunk_index * args.episode_chunk_size
        chunk_end = min(chunk_start + args.episode_chunk_size, len(selected_episode_ids))
        if chunk_start >= len(selected_episode_ids):
            raise IndexError(
                f"episode chunk {args.episode_chunk_index} is out of range for "
                f"{len(selected_episode_ids)} selected episodes"
            )
        selected_episode_ids = selected_episode_ids[chunk_start:chunk_end]
        print(
            f"[EpisodeChunk] index={args.episode_chunk_index} size={args.episode_chunk_size} "
            f"episodes={len(selected_episode_ids)} range=[{chunk_start}, {chunk_end})"
        )

    include_caption = args.model_kind == "support_caption"
    if include_caption and mode in {"uniform", "full"}:
        effective_manifest, chosen_records = build_first_demo_manifest(
            raw_records,
            selected_episode_ids,
            output_dir / "first_demo_eval_manifest.jsonl",
        )
        support_records = list(chosen_records.values())
        support_rounds = 1
    elif include_caption:
        effective_manifest = Path(args.raw_support_manifest)
        support_records = raw_records
        support_rounds = args.support_rounds_per_cycle
    else:
        effective_manifest = None
        support_records = []
        support_rounds = 1

    cfg, data_config, base_dataset, norm_stats_path = prepare_config_and_data(
        args,
        manifest_path=effective_manifest,
        support_rounds_per_cycle=support_rounds,
    )

    replay_seed = cfg.seed
    training_episode_ids: tuple[int, ...] | None = None
    if mode == "trainlike":
        training_episode_ids = select_episode_ids(
            episode_info,
            episode_order,
            raw_records,
            task_names=None,
            task_config="all",
            data_scope=args.data_scope,
        )
        train_steps = args.train_steps
        if train_steps is None:
            if not str(args.step).isdigit():
                raise ValueError("--train-steps is required when --step is not numeric")
            train_steps = int(args.step)
        replay_file = args.replay_file
        if replay_file is None:
            task_label = "all" if task_names is None else "-".join(task_names)
            replay_file = (
                output_dir.parent
                / "replay"
                / f"caption_{args.data_scope}_{task_label}_{args.task_config}_steps{train_steps}_bs{args.train_batch_size}.npz"
            )
        args.train_steps = train_steps
        args.replay_file = str(replay_file)
        replay_seed = cfg.seed if args.replay_seed is None else args.replay_seed

    selection_context = _shared_selection_context(
        args,
        mode=mode,
        selected_episode_ids=selected_episode_ids,
        dataset_length=len(base_dataset),
        replay_seed=replay_seed,
    )
    selection_path = Path(args.selection_file) if args.selection_file is not None else None
    if selection_path is not None and selection_path.is_file():
        selection, selection_digest = load_shared_selection(selection_path, selection_context)
    else:
        if selection_path is not None and args.require_existing_selection:
            raise FileNotFoundError(f"Required shared selection not found: {selection_path}")
        if mode == "uniform":
            selection = build_uniform_selection(
                base_dataset,
                selected_episode_ids,
                samples_per_episode=args.samples_per_episode,
            )
        elif mode == "full":
            selection = build_full_selection(base_dataset, selected_episode_ids)
        else:
            assert training_episode_ids is not None
            selection = build_or_load_trainlike_selection(
                base_dataset,
                training_episode_ids,
                selected_episode_ids,
                args.replay_file,
                train_steps=args.train_steps,
                train_batch_size=args.train_batch_size,
                support_rounds_per_cycle=args.support_rounds_per_cycle,
                eval_samples=args.eval_samples,
                replay_seed=replay_seed,
                sample_method=args.replay_sample_method,
                sample_seed=args.sample_seed,
                force_remake=args.force_remake_replay,
            )
        selection_digest = selection_sha256(selection, selection_context)
        if selection_path is not None:
            selection_digest = save_shared_selection(selection_path, selection, selection_context)

    validate_selection(selection, base_dataset, selected_episode_ids)

    evaluate_selection(
        args=args,
        mode=mode,
        cfg=cfg,
        data_config=data_config,
        base_dataset=base_dataset,
        selection=selection,
        selection_digest=selection_digest,
        norm_stats_path=norm_stats_path,
        episode_info=episode_info,
        support_records=support_records,
    )
