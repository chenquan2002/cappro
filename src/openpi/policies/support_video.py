"""Inference-time loading of human support videos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.models import model as _model


def attach_video_support(
    observation: dict,
    support_context: dict[str, np.ndarray],
    *,
    chunk_progress: float,
) -> dict:
    """Attach only video support fields to one inference observation."""
    expected_fields = {
        "support_images",
        "support_image_mask",
        "support_frame_progress",
    }
    unexpected_fields = set(support_context) - expected_fields
    missing_fields = expected_fields - set(support_context)
    if unexpected_fields or missing_fields:
        raise ValueError(
            f"Video support fields mismatch: missing={sorted(missing_fields)}, unexpected={sorted(unexpected_fields)}"
        )

    result = dict(observation)
    result.update(support_context)
    result["chunk_progress"] = np.asarray(
        [np.clip(float(chunk_progress), 0.0, 1.0)],
        dtype=np.float32,
    )
    return result


def make_null_video_support(*, num_frames: int) -> dict[str, np.ndarray]:
    """Return a masked-out support video context for support-ablation inference."""
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    return {
        "support_images": np.zeros((num_frames, *_model.IMAGE_RESOLUTION, 3), dtype=np.uint8),
        "support_image_mask": np.zeros((num_frames,), dtype=bool),
        "support_frame_progress": np.zeros((num_frames,), dtype=np.float32),
    }


class SupportVideoBank:
    """Load fixed-size support videos from the RoboTwin human-video bank."""

    def __init__(self, root: str | Path, *, num_frames: int = 8):
        self.root = Path(root)
        self.num_frames = int(num_frames)
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")

    def _task_root(self, task_name: str, task_config: str) -> Path:
        task_root = self.root / "human" / task_name / task_config
        if not task_root.is_dir():
            raise FileNotFoundError(f"Support task root not found: {task_root}")
        return task_root

    def discover(self, task_name: str, task_config: str) -> tuple[tuple[str, str], ...]:
        """Return all available ``(support_id, view)`` pairs for one task."""
        task_root = self._task_root(task_name, task_config)
        candidates = tuple(
            (frames_path.parent.parent.name, frames_path.parent.name)
            for frames_path in sorted(task_root.glob("*/*/frames.npy"))
        )
        if not candidates:
            raise FileNotFoundError(f"No support frames.npy found under: {task_root}")
        return candidates

    def load(self, task_name: str, task_config: str, support_id: str, view: str) -> dict[str, np.ndarray]:
        view_dir = self._task_root(task_name, task_config) / support_id / view
        frames_path = view_dir / "frames.npy"
        meta_path = view_dir / "meta.json"
        if not frames_path.is_file():
            raise FileNotFoundError(f"Support frames not found: {frames_path}")

        frames = np.load(frames_path, allow_pickle=False)
        expected_shape = (self.num_frames, *_model.IMAGE_RESOLUTION, 3)
        if frames.dtype != np.uint8:
            raise ValueError(f"Expected uint8 support frames, got {frames.dtype}: {frames_path}")
        if frames.shape != expected_shape:
            raise ValueError(f"Expected support frames shape {expected_shape}, got {frames.shape}: {frames_path}")

        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            progress = np.asarray(metadata.get("progress", []), dtype=np.float32)
        else:
            progress = np.linspace(0.0, 1.0, self.num_frames, dtype=np.float32)
        if progress.shape != (self.num_frames,):
            raise ValueError(f"Expected support progress shape ({self.num_frames},), got {progress.shape}: {meta_path}")
        if not np.all(np.isfinite(progress)) or np.any(progress < 0.0) or np.any(progress > 1.0):
            raise ValueError(f"Support progress must be finite and within [0, 1]: {meta_path}")

        return {
            "support_images": frames,
            "support_image_mask": np.ones((self.num_frames,), dtype=bool),
            "support_frame_progress": progress,
        }
