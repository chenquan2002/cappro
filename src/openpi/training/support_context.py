"""Manifest-backed support-video loading for pi0/pi0.5 training."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from itertools import pairwise
import json
import logging
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np

import openpi.models.tokenizer as _tokenizer

SUPPORT_VIEW_OVERRIDES = frozenset({"none", "ego", "front", "left", "right"})
CAPTION_PHASE_SCHEMA = "task_progress_v1"
logger = logging.getLogger("openpi")


def _normalize_support_view_override(value: str) -> str:
    view = str(value).strip().lower()
    if view not in SUPPORT_VIEW_OVERRIDES:
        raise ValueError(
            f"support_view_override must be one of {sorted(SUPPORT_VIEW_OVERRIDES)}, got {value!r}"
        )
    return view


def _validate_phase_caption_fields(record: dict[str, Any]) -> None:
    phase_keys = {
        "caption_phase_schema",
        "caption_phase_boundaries",
        "video_phase_captions",
    }
    present_keys = phase_keys.intersection(record)
    if not present_keys:
        return
    if present_keys != phase_keys:
        missing = sorted(phase_keys - present_keys)
        raise ValueError(f"Incomplete phase caption fields; missing {missing}")
    if record["caption_phase_schema"] != CAPTION_PHASE_SCHEMA:
        raise ValueError(
            f"Unsupported caption_phase_schema={record['caption_phase_schema']!r}; "
            f"expected {CAPTION_PHASE_SCHEMA!r}"
        )

    raw_boundaries = record["caption_phase_boundaries"]
    raw_captions = record["video_phase_captions"]
    if not isinstance(raw_boundaries, list) or not isinstance(raw_captions, list):
        raise ValueError("caption_phase_boundaries and video_phase_captions must be JSON arrays")
    if not raw_captions:
        raise ValueError("video_phase_captions must not be empty")
    if len(raw_captions) != len(raw_boundaries) + 1:
        raise ValueError(
            "video_phase_captions must contain exactly one more item than caption_phase_boundaries"
        )

    boundaries: list[float] = []
    for value in raw_boundaries:
        if isinstance(value, bool):
            raise ValueError("caption phase boundaries must be finite numbers in (0, 1)")
        try:
            boundary = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("caption phase boundaries must be finite numbers in (0, 1)") from error
        if not np.isfinite(boundary) or not 0.0 < boundary < 1.0:
            raise ValueError("caption phase boundaries must be finite numbers in (0, 1)")
        boundaries.append(boundary)
    if any(left >= right for left, right in pairwise(boundaries)):
        raise ValueError("caption phase boundaries must be strictly increasing")

    if any(not isinstance(caption, str) for caption in raw_captions):
        raise ValueError("video_phase_captions must contain only non-empty strings")
    captions = tuple(caption.strip() for caption in raw_captions)
    if any(not caption for caption in captions):
        raise ValueError("video_phase_captions must contain only non-empty strings")

    # Normalize once when loading the manifest so sample access only performs a
    # binary search and does not repeatedly parse phase metadata.
    record["caption_phase_boundaries"] = tuple(boundaries)
    record["video_phase_captions"] = captions


def select_caption_for_progress(record: dict[str, Any], progress: float) -> str:
    """Select phase supervision, falling back to the legacy full-video caption."""
    captions = record.get("video_phase_captions")
    if captions is None:
        return str(record.get("video_caption", "")).strip()

    progress = float(progress)
    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError(f"Caption selection progress must be in [0, 1], got {progress}")
    boundaries = record["caption_phase_boundaries"]
    return str(captions[bisect.bisect_right(boundaries, progress)]).strip()


class SupportManifest:
    """Lookup table for ``(global_episode_index, support_round_id)``."""

    def __init__(
        self,
        support_manifest_path: str | Path,
        *,
        support_view_override: str = "none",
        num_support_frames: int = 8,
    ):
        self.path = Path(support_manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Support manifest not found: {self.path}")
        self.support_view_override = _normalize_support_view_override(support_view_override)
        self.num_support_frames = int(num_support_frames)
        override_metadata: dict[Path, tuple[str, list[float]] | str] = {}

        self._records: dict[tuple[int, int], dict[str, Any]] = {}
        self._episode_lengths: dict[int, int] = {}
        round_ids: dict[int, list[int]] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    _validate_phase_caption_fields(record)
                except ValueError as error:
                    raise ValueError(f"Invalid phase caption metadata at {self.path}:{line_number}: {error}") from error
                if self.support_view_override != "none" and record.get("support_type") != "null":
                    record = self._override_support_view(record, override_metadata)
                episode_index = int(record["global_episode_index"])
                round_id = int(record.get("support_round_id", 0))
                key = (episode_index, round_id)
                if key in self._records:
                    raise ValueError(f"Duplicate support manifest key: {key}")
                self._records[key] = record
                round_ids.setdefault(episode_index, []).append(round_id)
                if "episode_length" in record:
                    episode_length = int(record["episode_length"])
                    previous = self._episode_lengths.setdefault(episode_index, episode_length)
                    if previous != episode_length:
                        raise ValueError(
                            f"Inconsistent episode_length for global_episode_index={episode_index}: "
                            f"{previous} != {episode_length}"
                        )

        if not self._records:
            raise ValueError(f"Support manifest is empty: {self.path}")
        self._round_ids = {episode_index: tuple(sorted(ids)) for episode_index, ids in round_ids.items()}

    def _override_support_view(
        self,
        record: dict[str, Any],
        metadata_cache: dict[Path, tuple[str, list[float]] | str],
    ) -> dict[str, Any]:
        frames_value = str(record.get("support_frames_npy", "")).strip()
        if not frames_value or not bool(record.get("has_support", True)):
            return record

        original_frames_path = Path(frames_value)
        target_dir = original_frames_path.parent.parent / self.support_view_override
        target_frames_path = target_dir / "frames.npy"
        target_meta_path = target_dir / "meta.json"
        if target_meta_path not in metadata_cache:
            try:
                frames = np.load(target_frames_path, allow_pickle=False, mmap_mode="r")
                expected_shape = (self.num_support_frames, 224, 224, 3)
                if frames.dtype != np.uint8 or frames.shape != expected_shape:
                    raise ValueError(
                        f"expected {expected_shape} uint8 frames, got {frames.shape} {frames.dtype}"
                    )
                metadata = json.loads(target_meta_path.read_text(encoding="utf-8"))
                progress = np.asarray(metadata.get("progress"), dtype=np.float32)
                if (
                    progress.shape != (self.num_support_frames,)
                    or not np.all(np.isfinite(progress))
                    or np.any(progress < 0.0)
                    or np.any(progress > 1.0)
                ):
                    raise ValueError(f"invalid progress: {progress.tolist()}")
                metadata_cache[target_meta_path] = (str(target_frames_path), progress.tolist())
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                reason = f"{type(error).__name__}: {error}"
                metadata_cache[target_meta_path] = reason
                logger.warning(
                    "Skipping unavailable support view %s at %s: %s",
                    self.support_view_override,
                    target_dir,
                    reason,
                )

        target_metadata = metadata_cache[target_meta_path]
        if isinstance(target_metadata, str):
            mapped = dict(record)
            mapped["support_original_view"] = record.get("support_view")
            mapped["support_view"] = self.support_view_override
            mapped["support_type"] = "null"
            mapped["has_support"] = False
            mapped["support_frames_npy"] = ""
            mapped["support_frame_progress"] = [0.0] * self.num_support_frames
            mapped["support_skip_reason"] = target_metadata
            return mapped

        target_frames, target_progress = target_metadata
        mapped = dict(record)
        mapped["support_original_view"] = record.get("support_view")
        mapped["support_view"] = self.support_view_override
        mapped["support_frames_npy"] = target_frames
        mapped["support_frame_progress"] = target_progress
        return mapped

    def get(self, global_episode_index: int, support_round_id: int) -> dict[str, Any]:
        key = (int(global_episode_index), int(support_round_id))
        try:
            return self._records[key]
        except KeyError as error:
            raise KeyError(
                f"No support record for global_episode_index={key[0]}, support_round_id={key[1]}. "
                f"Manifest={self.path}"
            ) from error

    def episode_length(self, global_episode_index: int) -> int:
        episode_index = int(global_episode_index)
        try:
            return self._episode_lengths[episode_index]
        except KeyError as error:
            raise KeyError(f"No episode_length for global_episode_index={episode_index} in {self.path}") from error

    def round_ids(self, global_episode_index: int) -> tuple[int, ...]:
        episode_index = int(global_episode_index)
        try:
            return self._round_ids[episode_index]
        except KeyError as error:
            raise KeyError(f"No support rounds for global_episode_index={episode_index} in {self.path}") from error


class SupportFrameCache:
    """Small LRU cache for uint8 ``frames.npy`` arrays."""

    def __init__(self, max_items: int = 1024):
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, frames_npy: str | Path) -> np.ndarray:
        path = str(frames_npy)
        if path in self._cache:
            value = self._cache.pop(path)
            self._cache[path] = value
            return value

        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 support frames, got {arr.dtype}: {path}")
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected support frames shape [K,H,W,3], got {arr.shape}: {path}")

        if self.max_items > 0:
            while len(self._cache) >= self.max_items:
                self._cache.popitem(last=False)
            self._cache[path] = arr
        return arr


class SupportRoundDataset:
    """Logically pair every base sample with multiple manifest support rounds."""

    def __init__(self, dataset, support_rounds_per_cycle: int):
        if support_rounds_per_cycle < 1:
            raise ValueError("support_rounds_per_cycle must be >= 1")
        self._dataset = dataset
        self._base_len = len(dataset)
        self._rounds = int(support_rounds_per_cycle)

    def __len__(self) -> int:
        return self._base_len * self._rounds

    def __getitem__(self, index: SupportsIndex):
        index = int(index.__index__() if hasattr(index, "__index__") else index)
        base_index = index % self._base_len
        support_round_id = index // self._base_len
        item = dict(self._dataset[base_index])
        item["support_round_id"] = np.asarray(support_round_id, dtype=np.int64)
        return item


class AddSupportContext:
    """Attach support frames and train-only caption supervision to one robot sample."""

    _CAPTION_FIELDS = (
        "caption_input_tokens",
        "caption_input_mask",
        "caption_target_tokens",
        "caption_loss_mask",
        "caption_hand_side_mask",
    )

    def __init__(
        self,
        support_manifest_path: str | Path,
        *,
        num_support_frames: int = 8,
        support_cache_size: int = 1024,
        caption_max_len: int = 128,
        support_chunk_size: int = 1,
        support_view_override: str = "none",
        include_caption_supervision: bool = True,
    ):
        self.manifest = SupportManifest(
            support_manifest_path,
            support_view_override=support_view_override,
            num_support_frames=num_support_frames,
        )
        self.cache = SupportFrameCache(max_items=support_cache_size)
        self.num_support_frames = int(num_support_frames)
        self.caption_max_len = int(caption_max_len)
        self.support_chunk_size = int(support_chunk_size)
        self.include_caption_supervision = bool(include_caption_supervision)
        self._tokenizer: _tokenizer.PaligemmaTokenizer | None = None
        self._runtime_invalid_frames: set[str] = set()

    def _caption_tokenizer(self) -> _tokenizer.PaligemmaTokenizer:
        if self._tokenizer is None:
            self._tokenizer = _tokenizer.PaligemmaTokenizer(self.caption_max_len)
        return self._tokenizer

    @staticmethod
    def _as_int(value: Any) -> int:
        return int(np.asarray(value).reshape(-1)[0])

    def _add_empty_caption(self, data: dict[str, Any]) -> None:
        if not self.include_caption_supervision:
            return
        data["caption_input_tokens"] = np.zeros((self.caption_max_len,), dtype=np.int32)
        data["caption_input_mask"] = np.zeros((self.caption_max_len,), dtype=bool)
        data["caption_target_tokens"] = np.zeros((self.caption_max_len,), dtype=np.int32)
        data["caption_loss_mask"] = np.zeros((self.caption_max_len,), dtype=bool)
        data["caption_hand_side_mask"] = np.zeros((self.caption_max_len,), dtype=bool)

    def _add_caption(self, data: dict[str, Any], caption: str) -> None:
        if not self.include_caption_supervision:
            return
        values = self._caption_tokenizer().tokenize_caption_teacher_forcing(caption)
        data.update(dict(zip(self._CAPTION_FIELDS, values, strict=True)))

    def _add_null_support(self, data: dict[str, Any], chunk_progress: float) -> dict[str, Any]:
        data["support_images"] = np.zeros(
            (self.num_support_frames, 224, 224, 3),
            dtype=np.uint8,
        )
        data["support_image_mask"] = np.zeros((self.num_support_frames,), dtype=bool)
        data["support_frame_progress"] = np.zeros((self.num_support_frames,), dtype=np.float32)
        data["chunk_progress"] = np.asarray([chunk_progress], dtype=np.float32)
        self._add_empty_caption(data)
        return data

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "episode_index" not in data or "frame_index" not in data:
            raise KeyError(
                "AddSupportContext requires episode_index and frame_index. "
                "Check that repack and policy transforms preserve this metadata."
            )

        data = dict(data)
        global_episode_index = self._as_int(data["episode_index"])
        frame_index = self._as_int(data["frame_index"])
        sample_round_id = self._as_int(data.get("support_round_id", 0))
        episode_length = self.manifest.episode_length(global_episode_index)
        round_ids = self.manifest.round_ids(global_episode_index)

        chunk_index = 0 if self.support_chunk_size <= 1 else frame_index // self.support_chunk_size
        round_index = (sample_round_id + chunk_index) % len(round_ids)
        support = self.manifest.get(global_episode_index, round_ids[round_index])
        chunk_progress = frame_index / max(episode_length - 1, 1)

        has_support = bool(support.get("has_support", True))
        frames_path = str(support.get("support_frames_npy", "")).strip()
        if not has_support or support.get("support_type") == "null" or not frames_path:
            return self._add_null_support(data, chunk_progress)
        if frames_path in self._runtime_invalid_frames:
            return self._add_null_support(data, chunk_progress)

        try:
            frames = self.cache.get(frames_path)
        except (OSError, ValueError) as error:
            if frames_path not in self._runtime_invalid_frames:
                self._runtime_invalid_frames.add(frames_path)
                logger.warning("Skipping unreadable support frames %s: %s", frames_path, error)
            return self._add_null_support(data, chunk_progress)
        if frames.shape[0] != self.num_support_frames:
            if frames_path not in self._runtime_invalid_frames:
                self._runtime_invalid_frames.add(frames_path)
                logger.warning(
                    "Skipping support frames with count %s, expected %s: %s",
                    frames.shape[0],
                    self.num_support_frames,
                    frames_path,
                )
            return self._add_null_support(data, chunk_progress)
        try:
            progress = np.asarray(support["support_frame_progress"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            self._runtime_invalid_frames.add(frames_path)
            logger.warning("Skipping support with invalid progress at %s: %s", frames_path, error)
            return self._add_null_support(data, chunk_progress)
        if progress.shape != (self.num_support_frames,) or not np.all(np.isfinite(progress)):
            self._runtime_invalid_frames.add(frames_path)
            logger.warning("Skipping support with invalid progress shape or values at %s", frames_path)
            return self._add_null_support(data, chunk_progress)

        data["support_images"] = frames
        data["support_image_mask"] = np.ones((self.num_support_frames,), dtype=bool)
        data["support_frame_progress"] = progress
        data["chunk_progress"] = np.asarray([chunk_progress], dtype=np.float32)

        if self.include_caption_supervision:
            caption = select_caption_for_progress(support, chunk_progress)
            if not caption:
                raise ValueError(
                    f"Missing caption supervision for episode={global_episode_index}, round={round_ids[round_index]}"
                )
            self._add_caption(data, caption)
        return data
