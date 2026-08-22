#!/usr/bin/env python3
"""Build a support manifest for human-video in-context training.

The manifest is generated at data-processing time. Training only does lookup by:
  (global_episode_index, support_round_id)

This solves two issues:
  1. LeRobot mixed repos contain many tasks, so global episode_index must be mapped
     through meta/episode_origin.jsonl.
  2. If training uses fewer rounds than available videos, each trajectory still sees
     a deterministic shuffled subset rather than always demo_000, demo_001, ...

Supported view modes:
  center/left/right/ego: candidates are M demos with that fixed view.
  random: candidates are all M demos × 4 views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

VIEWS_ALL = ("left", "front", "right", "ego")
FIXED_VIEW_MODES = set(VIEWS_ALL)
SUPPORT_TEXT_MODES = (
    "none",
    "video_instruction",
    "video_caption",
    "video_instruction_caption",
)


def read_support_info(demo_dir: Path, strict: bool = True) -> dict[str, str]:
    """ICL SUPPORT:
    Read structured support metadata from support_bank.

    This version requires support_info.json and does not use caption.txt.
    """
    info_path = demo_dir / "support_info.json"

    if not info_path.exists():
        if strict:
            raise FileNotFoundError(f"Missing support_info.json: {info_path}")
        return {
            "video_instruction": "",
            "video_caption": "",
            "video_task_name": "",
            "support_relation": "",
            "support_info_path": "",
        }

    info = json.loads(info_path.read_text(encoding="utf-8"))

    return {
        "video_instruction": str(info.get("video_instruction", "")).strip(),
        "video_caption": str(info.get("video_caption", "")).strip(),
        "video_task_name": str(info.get("video_task_name", "")).strip(),
        "support_relation": str(info.get("support_relation", "")).strip(),
        "support_info_path": str(info_path),
    }


def build_support_text(info: dict[str, str], mode: str) -> str:
    """ICL SUPPORT:
    Construct the actual support text that enters the model.

    The output is still stored as `support_caption` in manifest so that
    model.py / pi0.py do not need to change.
    """
    if mode == "none":
        return ""

    if mode == "video_instruction":
        instruction = info.get("video_instruction", "")
        return f"[VIDEO TASK] {instruction}".strip() if instruction else ""

    if mode == "video_caption":
        caption = info.get("video_caption", "")
        return f"[VIDEO CAPTION] {caption}".strip() if caption else ""

    if mode == "video_instruction_caption":
        instruction = info.get("video_instruction", "")
        caption = info.get("video_caption", "")

        parts = []
        if instruction:
            parts.append(f"[VIDEO TASK] {instruction}")
        if caption:
            parts.append(f"[VIDEO CAPTION] {caption}")
        return "\n".join(parts)

    raise ValueError(f"Unsupported support_text_mode={mode}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_u32(*parts: Any, seed: int = 0) -> int:
    s = "||".join(str(p) for p in parts) + f"||seed={seed}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def candidate_sort_key(c: dict[str, Any]) -> tuple[int, str]:
    # human_demo_003 -> 3
    sid = c["support_id"]
    try:
        n = int(str(sid).split("_")[-1])
    except Exception:
        n = 10**9
    return n, c["support_view"]


def build_candidates(
    *,
    support_bank_root: Path,
    task_name: str,
    task_config: str,
    view_mode: str,
    num_human_demos: int | None,
    support_text_mode: str,
) -> list[dict[str, Any]]:
    root = support_bank_root / "human" / task_name / task_config
    if not root.exists():
        print(f"[WARN] Missing support bank task/config root: {root}")
        return []

    if view_mode == "random":
        views = list(VIEWS_ALL)
    elif view_mode in FIXED_VIEW_MODES:
        views = [view_mode]
    else:
        raise ValueError(f"Unsupported view_mode={view_mode}; use {sorted(FIXED_VIEW_MODES | {'random'})}")

    demo_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("human_demo_")])
    if num_human_demos is not None:
        expected = {f"human_demo_{i:03d}" for i in range(num_human_demos)}
        demo_dirs = [p for p in demo_dirs if p.name in expected]

    candidates: list[dict[str, Any]] = []
    for demo_dir in demo_dirs:
        support_info = read_support_info(demo_dir, strict=True)
        support_text = build_support_text(support_info, support_text_mode)
        for view in views:
            view_dir = demo_dir / view
            meta_path = view_dir / "meta.json"
            frames_npy = view_dir / "frames.npy"
            if not meta_path.exists() or not frames_npy.exists():
                continue
            meta = read_json(meta_path)
            candidates.append(
                {
                    "support_type": "human",
                    "support_id": demo_dir.name,
                    "support_view": view,
                    "support_frames_npy": str(frames_npy),
                    "support_caption": support_text,
                    "support_text_mode": support_text_mode,
                    "support_frame_progress": meta.get("progress", []),
                    "video_instruction": support_info["video_instruction"],
                    "video_caption": support_info["video_caption"],
                    "video_task_name": support_info["video_task_name"],
                    "support_relation": support_info["support_relation"],
                    "support_info_path": support_info["support_info_path"],
                }
            )
    candidates.sort(key=candidate_sort_key)
    if not candidates:
        print(f"[WARN] No support candidates for {task_name}/{task_config}, view_mode={view_mode}, root={root}")
        return []
    return candidates


def default_rounds(view_mode: str, candidates: list[dict[str, Any]]) -> int:
    # If view=random, candidates are demo-view pairs. Otherwise candidates are demos.
    return len(candidates)

################modified
def make_null_support_entry(
    *,
    num_support_frames: int = 8,
) -> dict[str, Any]:
    """Create a null/placeholder support entry for tasks without human video.

    Training code will recognize this and set support_image_mask to all False.
    """
    return {
        "support_type": "null",
        "support_id": "null_support",
        "support_view": "none",
        "support_frames_npy": "",
        "support_caption": "",
        "support_text_mode": "none",
        "support_frame_progress": [0.0] * num_support_frames,
        "video_instruction": "",
        "video_caption": "",
        "video_task_name": "",
        "support_relation": "none",
        "support_info_path": "",
        "has_support": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-repo", required=True, type=Path, help="Path containing meta/episode_origin.jsonl")
    parser.add_argument("--support-bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--view-mode", default="center", choices=["front", "left", "right", "ego", "random"])
    parser.add_argument("--num-human-demos", type=int, default=10)
    parser.add_argument("--num-support-rounds", type=int, default=None,
                        help="If omitted, use number of candidates: 10 for fixed view, 40 for random if M=10.")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--support-text-mode",default="video_instruction_caption", choices=[
        "none",
        "video_instruction",
        "video_caption",
        "video_instruction_caption",
    ],
    help="How to construct support text from support_info.json.",
)
    args = parser.parse_args()

    origin_path = args.lerobot_repo / "meta" / "episode_origin.jsonl"
    if not origin_path.exists():
        raise FileNotFoundError(f"Missing episode origin: {origin_path}")
    origins = load_jsonl(origin_path)

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for origin in origins:
        task_name = origin["task_name"]
        task_config = origin["task_config"]
        key = (task_name, task_config)
        if key not in cache:
            cache[key] = build_candidates(
                support_bank_root=args.support_bank_root,
                task_name=task_name,
                task_config=task_config,
                view_mode=args.view_mode,
                num_human_demos=args.num_human_demos,
                support_text_mode=args.support_text_mode,
            )
            print(f"[INFO] {task_name}/{task_config}: {len(cache[key])} candidates for view_mode={args.view_mode}")
        candidates = cache[key]

        # If this task/config has no human support video, create one null-support
        # round. AddSupportContext will convert this into masked support inputs.
        if not candidates:
            print(f"[WARN] Using null support for {task_name}/{task_config}")
            candidates = [make_null_support_entry(num_support_frames=8)]
            rounds = 1
        else:
            rounds = args.num_support_rounds or default_rounds(args.view_mode, candidates)

        # Episode-specific deterministic permutation. Even if training uses only early rounds,
        # different trajectories will use different demo/view subsets.
        rng_seed = stable_u32(
            task_name,
            task_config,
            origin.get("source_episode_index", origin.get("local_episode_index", 0)),
            origin["global_episode_index"],
            seed=args.shuffle_seed,
        )
        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(len(candidates)).tolist()

        for round_id in range(rounds):
            c = candidates[perm[round_id % len(candidates)]]
            record = {
                "global_episode_index": int(origin["global_episode_index"]),
                "support_round_id": int(round_id),
                "task_name": task_name,
                "task_config": task_config,
                "local_episode_index": int(origin.get("local_episode_index", -1)),
                "source_episode_index": int(origin.get("source_episode_index", -1)),
                "episode_length": int(origin["episode_length"]),
                "view_mode": args.view_mode,
                **c,
            }
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(records)} support records: {args.output}")


if __name__ == "__main__":
    main()
