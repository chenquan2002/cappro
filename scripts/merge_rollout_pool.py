#!/usr/bin/env python3
"""Merge round dataset into the current pool dataset.

This script is an orchestrator around LeRobot dataset merge semantics used by ACP:
- round 0: pool = round dataset
- round k>0: pool_k = merge(pool_{k-1}, round_k)

It does not change sampling ratios or dataset contents; it only constructs the
next pool dataset path and invokes a dataset merge command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


_ROBOTWIN_ROOT = Path(__file__).resolve().parents[3]


def infer_repo_root(repo_id: str) -> Path:
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return xdg_cache / "huggingface" / "lerobot" / repo_id


def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_round_to_pool(round_path: Path, pool_path: Path, overwrite: bool) -> None:
    if pool_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Pool dataset already exists: {pool_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(pool_path)
    ensure_parent(pool_path)
    shutil.copytree(round_path, pool_path)


def remove_target_if_needed(target_path: Path, overwrite: bool, *, desc: str) -> None:
    if not target_path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{desc} already exists: {target_path}. Use --overwrite to replace it."
        )
    shutil.rmtree(target_path)


def run_merge_command(
    pool_prev_repo_id: str,
    round_repo_id: str,
    pool_repo_id: str,
    dry_run: bool,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Invoke lerobot dataset merge.

    The exact CLI binary may vary by environment, so we try a few common forms.
    """
    candidate_cmds = [
        [
            "lerobot-edit-dataset",
            f"--source-repo-ids={pool_prev_repo_id},{round_repo_id}",
            f"--target-repo-id={pool_repo_id}",
            "--operation.type=merge",
        ],
        [
            "uv",
            "run",
            "lerobot-edit-dataset",
            f"--source-repo-ids={pool_prev_repo_id},{round_repo_id}",
            f"--target-repo-id={pool_repo_id}",
            "--operation.type=merge",
        ],
        [
            "python",
            "-m",
            "lerobot.scripts.lerobot_edit_dataset",
            f"--source-repo-ids={pool_prev_repo_id},{round_repo_id}",
            f"--target-repo-id={pool_repo_id}",
            "--operation.type=merge",
        ],
    ]

    if dry_run:
        print("[dry-run] Would run one of the following merge commands:")
        for cmd in candidate_cmds:
            print("  ", " ".join(cmd))
        return

    last_err: Exception | None = None
    for cmd in candidate_cmds:
        try:
            print("[merge] Trying:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)
            print("[merge] Success")
            return
        except Exception as exc:
            last_err = exc
            print(f"[merge] Failed with command: {' '.join(cmd)}", file=sys.stderr)
    raise RuntimeError(
        "Failed to merge datasets with all known lerobot-edit-dataset command forms. "
        "Please verify your LeRobot installation and CLI availability."
    ) from last_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge current round dataset into pool dataset")
    parser.add_argument("--run_name", required=True, help="Experiment run name")
    parser.add_argument("--round_id", required=True, type=int, help="Current round id")
    parser.add_argument("--repo_id_round", required=True, help="Repo id of current round dataset")
    parser.add_argument("--repo_id_pool", required=True, help="Repo id of target pool dataset")
    parser.add_argument(
        "--repo_id_pool_prev",
        default=None,
        help="Repo id of previous pool dataset. Required when round_id > 0 unless --pool_prev_path is given.",
    )
    parser.add_argument(
        "--round_path",
        default=None,
        help="Optional explicit path to current round dataset. Defaults to XDG/HF cache path inferred from repo_id_round.",
    )
    parser.add_argument(
        "--pool_path",
        default=None,
        help="Optional explicit path to target pool dataset. Defaults to XDG/HF cache path inferred from repo_id_pool.",
    )
    parser.add_argument(
        "--pool_prev_path",
        default=None,
        help="Optional explicit path to previous pool dataset. Defaults to XDG/HF cache path inferred from repo_id_pool_prev.",
    )
    parser.add_argument(
        "--experiment_root",
        default=str(_ROBOTWIN_ROOT / "policy" / "experiments"),
        help="Root directory for experiment metadata/log files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target pool path")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without executing them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if not xdg_cache:
        print(
            "[warn] XDG_CACHE_HOME is not set. Falling back to ~/.cache. "
            "If your datasets are on NAS, export XDG_CACHE_HOME first.",
            file=sys.stderr,
        )

    round_path = Path(args.round_path) if args.round_path else infer_repo_root(args.repo_id_round)
    pool_path = Path(args.pool_path) if args.pool_path else infer_repo_root(args.repo_id_pool)

    ensure_exists(round_path, "Current round dataset")

    experiment_round_root = Path(args.experiment_root) / args.run_name / f"round_{args.round_id}"
    experiment_round_root.mkdir(parents=True, exist_ok=True)
    meta_path = experiment_round_root / "pool_merge_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"run_name={args.run_name}\n")
        f.write(f"round_id={args.round_id}\n")
        f.write(f"repo_id_round={args.repo_id_round}\n")
        f.write(f"repo_id_pool={args.repo_id_pool}\n")
        f.write(f"round_path={round_path}\n")
        f.write(f"pool_path={pool_path}\n")
        f.write(f"repo_id_pool_prev={args.repo_id_pool_prev}\n")
        f.write(f"pool_prev_path={args.pool_prev_path}\n")
        f.write(f"experiment_root={Path(args.experiment_root).resolve()}\n")
        f.write(f"xdg_cache_home={os.environ.get('XDG_CACHE_HOME', '')}\n")

    if args.round_id == 0:
        if args.dry_run:
            print(f"[dry-run] Would initialize pool by copying:\n  {round_path}\n-> {pool_path}")
            return
        print(f"[pool] round 0: copy round dataset to pool\n  {round_path}\n-> {pool_path}")
        copy_round_to_pool(round_path, pool_path, overwrite=args.overwrite)
        print("[pool] round 0 pool initialized")
        return

    if args.pool_prev_path:
        pool_prev_path = Path(args.pool_prev_path)
        ensure_exists(pool_prev_path, "Previous pool dataset")
    else:
        if not args.repo_id_pool_prev:
            raise ValueError("For round_id > 0, provide --repo_id_pool_prev or --pool_prev_path")
        pool_prev_path = infer_repo_root(args.repo_id_pool_prev)
        ensure_exists(pool_prev_path, "Previous pool dataset")

    print("[pool] round > 0: merge previous pool with current round dataset")
    print("  prev pool:", pool_prev_path)
    print("  round    :", round_path)
    print("  target   :", pool_path)

    # Prevent accidental re-run confusion.
    remove_target_if_needed(pool_path, args.overwrite, desc="Target pool dataset")

    # The merge command works on repo ids, not paths; paths are only checked here for clarity.
    child_env = os.environ.copy()
    run_merge_command(
        args.repo_id_pool_prev,
        args.repo_id_round,
        args.repo_id_pool,
        dry_run=args.dry_run,
        env=child_env,
    )


if __name__ == "__main__":
    main()
