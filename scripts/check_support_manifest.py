#!/usr/bin/env python3
"""Validate human support manifest and support bank files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-support-rounds", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--check-npy-shape", action="store_true", help="Load every frames.npy and verify shape/dtype. Slower but safer.")
    args = parser.parse_args()

    origins = load_jsonl(args.lerobot_repo / "meta" / "episode_origin.jsonl")
    records = load_jsonl(args.manifest)
    expected_eps = {int(r["global_episode_index"]) for r in origins}

    by_ep = defaultdict(list)
    support_counter = Counter()
    view_counter = Counter()

    errors: list[str] = []
    null_support_count = 0
    for r in records:
        ep = int(r["global_episode_index"])
        round_id = int(r["support_round_id"])
        by_ep[ep].append(round_id)

        # Null support handling: skip file checks for tasks without human video.
        has_support = bool(r.get("has_support", True))
        support_type = str(r.get("support_type", "human"))
        is_null_support = not has_support or support_type == "null"
        if is_null_support:
            null_support_count += 1
            # For null support, only check progress length.
            progress = r.get("support_frame_progress", [])
            if len(progress) != args.num_frames:
                errors.append(f"bad progress length for null support ep={ep}, round={round_id}: {len(progress)}")
            # Null support should have empty/masked values.
            if r.get("support_frames_npy", "") != "":
                errors.append(f"null support should have empty support_frames_npy for ep={ep}")
            support_counter[(r.get("task_name"), r.get("task_config"), r.get("support_id"))] += 1
            view_counter[r.get("support_view")] += 1
            continue

        # Normal support video checks.
        npy_path = Path(r["support_frames_npy"])
        if not npy_path.exists():
            errors.append(f"missing frames.npy: {npy_path}")
        elif args.check_npy_shape:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.dtype != np.uint8:
                errors.append(f"bad dtype {arr.dtype}: {npy_path}")
            if arr.ndim != 4 or arr.shape[0] != args.num_frames or arr.shape[-1] != 3:
                errors.append(f"bad shape {arr.shape}: {npy_path}")

        progress = r.get("support_frame_progress", [])
        if len(progress) != args.num_frames:
            errors.append(f"bad progress length for ep={ep}, round={round_id}: {len(progress)}")
        if progress and (min(progress) < -1e-6 or max(progress) > 1 + 1e-6):
            errors.append(f"progress out of range for ep={ep}, round={round_id}: {progress}")

        support_text_mode = str(r.get("support_text_mode", "video_instruction_caption")).strip()
        support_caption = str(r.get("support_caption", "")).strip()

        if support_text_mode != "none" and not support_caption:
            errors.append(
                f"empty support_caption for ep={ep}, round={round_id}, "
                f"support_text_mode={support_text_mode}"
            )

        support_info_path = Path(str(r.get("support_info_path", "")))
        if not support_info_path.exists():
            errors.append(
                f"missing support_info_path for ep={ep}, round={round_id}: {support_info_path}"
            )

        for key in ("video_instruction", "video_caption", "video_task_name", "support_relation"):
            if key not in r:
                errors.append(f"missing metadata key `{key}` for ep={ep}, round={round_id}")

        support_counter[(r.get("task_name"), r.get("task_config"), r.get("support_id"))] += 1
        view_counter[r.get("support_view")] += 1

    missing_eps = expected_eps - set(by_ep)
    if missing_eps:
        errors.append(f"missing episodes in manifest: first={sorted(missing_eps)[:20]}, count={len(missing_eps)}")

    if args.num_support_rounds is not None:
        expected_rounds = set(range(args.num_support_rounds))
        for ep in expected_eps:
            got = set(by_ep.get(ep, []))
            if got != expected_rounds:
                errors.append(f"bad rounds for ep={ep}: got={sorted(got)}, expected={sorted(expected_rounds)}")
                if len(errors) > 50:
                    break

    print(f"[INFO] origins episodes: {len(expected_eps)}")
    print(f"[INFO] manifest records: {len(records)}")
    print(f"[INFO] null support records: {null_support_count}")
    print(f"[INFO] view counts: {dict(view_counter)}")
    print(f"[INFO] unique support ids: {len(support_counter)}")

    if errors:
        print("[ERROR] manifest validation failed:")
        for e in errors[:100]:
            print("  -", e)
        raise SystemExit(1)
    print("[OK] support manifest validation passed")


if __name__ == "__main__":
    main()
