#!/usr/bin/env python3
"""Paired local perturbation-recovery evaluation for RoboTwin ``place_fan``.

The script intentionally does not use SAPIEN ``pack_poses`` / ``unpack_poses``:
they do not restore the complete robot articulation, drive-target, contact, and
grasp state. Every rollout instead follows the same reproducible procedure:

    clean source seed -> reset shared task scene -> expert-action replay to one frame
    -> one small wrist-roll perturbation -> policy closed-loop rollout

The comparison is fixed to two caption 70k checkpoints plus one pi05 20W checkpoint:

* caption checkpoint with ego support video;
* the same caption checkpoint with null/masked support video;
* vanilla pi05 base checkpoint at exactly step 200000.

No normal evaluation file, checkpoint, manifest, or expert trajectory is
modified. The script writes only under ``--output-dir``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from contextlib import contextmanager, suppress
import csv
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import h5py
import jax
import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

CAPTION_STEP = 70_000
PI05_STEP = 200_000
TASK_NAME = "place_fan"
TASK_CONFIG = "demo_clean"
ACTION_DIM = 14
ACTION_HORIZON = 50
EXPERT_REFERENCE_SCHEMA_VERSION = 2
CONDITION_ORDER = ("caption_support", "caption_masked", "pi05_base")
CONDITION_LABELS = {
    "caption_support": "Caption + support video",
    "caption_masked": "Caption, support masked",
    "pi05_base": "pi05 base",
}
CONDITION_COLORS = {
    "caption_support": "#b63a3a",
    "caption_masked": "#2675a8",
    "pi05_base": "#326d48",
}
# The provided clean source set contains successful episodes, not simulator
# seeds 0..49.  The original collector advances its seed until an episode
# succeeds, so episode 0 is seed 1 (seed 0 is a left/Maroon scene that was not
# included).  These are recovered from the source episode arm/color metadata
# and RoboTwin's deterministic place_fan randomization sequence.
PLACE_FAN_SOURCE_SEEDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
)
_SHARED_RECOVERY_ENV: dict[str, Any] = {"env": None}


CAPTION_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = CAPTION_ROOT.parents[1]
for _path in (CAPTION_ROOT / "src", CAPTION_ROOT, ROBOTWIN_ROOT / "description" / "utils", ROBOTWIN_ROOT):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from pi_model import PI0  # noqa: E402


@dataclass(frozen=True)
class EpisodeRecord:
    global_episode_index: int
    local_episode_index: int
    source_episode_index: int
    episode_length: int
    hdf5_path: Path
    environment_seed: int = -1


@dataclass(frozen=True)
class ModelSpec:
    key: str
    train_config_name: str
    exp_name: str
    checkpoint_root: Path
    checkpoint_step: int
    use_support_context: bool
    mask_support_video: bool

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.train_config_name / self.exp_name / str(self.checkpoint_step)


def encode_obs(observation: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """The joint-control observation mapping used by caption/deploy_policy.py."""

    return (
        [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
        ],
        np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    )


def get_chunk_progress(task_env) -> float:
    """Match normal caption deployment: action count divided by eval step limit."""

    step_limit = int(getattr(task_env, "step_lim", 0) or 0)
    if step_limit <= 1:
        return 0.0
    return float(np.clip(float(getattr(task_env, "take_action_cnt", 0)) / (step_limit - 1), 0.0, 1.0))


def _parse_csv(value: str, *, cast, name: str) -> tuple:
    values = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CAPTION_ROOT / "results" / "perturbation_recovery" / "place_fan_20W",
        help="New experiment root. Existing source data and normal eval outputs remain untouched.",
    )
    parser.add_argument(
        "--episode-origin",
        type=Path,
        default=(
            ROBOTWIN_ROOT
            / "data/lerobot_data/huggingface/lerobot/source_data_hovapi_repo/meta/episode_origin.jsonl"
        ),
    )
    parser.add_argument("--episode-count", type=int, default=50)
    parser.add_argument(
        "--episode-source-indices",
        default=None,
        help="Optional comma-separated clean source ids in 0..49. Overrides --episode-count.",
    )
    parser.add_argument("--progresses", default="0.55,0.65,0.75")
    parser.add_argument("--perturb-degrees", default="0,-1,1,-2,2,-5,5")
    parser.add_argument(
        "--rollout-repeats",
        type=int,
        default=1,
        help="Independent policy samples per paired condition. Default formal scale is 50*3*7*3=3150 rollouts.",
    )
    parser.add_argument(
        "--max-policy-actions",
        type=int,
        default=None,
        help="Optional cap after perturbation. Omit to use normal task success / 400-action limit.",
    )
    parser.add_argument("--policy-chunk-size", type=int, default=ACTION_HORIZON)
    parser.add_argument("--policy-seed", type=int, default=20260816)
    parser.add_argument("--support-id", default="human_demo_000")
    parser.add_argument("--support-view", choices=["ego"], default="ego")
    parser.add_argument(
        "--support-bank-root",
        type=Path,
        default=ROBOTWIN_ROOT / "data/support_data/support_bank_full",
    )
    parser.add_argument("--caption-config", default="pi05_aloha_robotwin_cappro_lora")
    parser.add_argument("--caption-exp", default="cappro_source_v1")
    parser.add_argument("--caption-step", type=int, default=CAPTION_STEP)
    parser.add_argument("--caption-checkpoint-root", type=Path, default=CAPTION_ROOT / "checkpoints")
    parser.add_argument("--pi05-config", default="pi05_aloha_robotwin_lora")
    parser.add_argument("--pi05-exp", default="source_data_20W_h100")
    parser.add_argument("--pi05-step", type=int, default=PI05_STEP)
    parser.add_argument(
        "--pi05-checkpoint-root",
        type=Path,
        default=ROBOTWIN_ROOT / "policy/pi05/checkpoints",
        help="Must contain vanilla pi05 base at .../source_data_20W_h100/200000.",
    )
    parser.add_argument("--norm-asset-id", default="source_data_iclpi_repo")
    parser.add_argument("--replay-atol", type=float, default=3e-3)
    parser.add_argument("--strict-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--planner-mode",
        choices=["joint_qpos", "normal"],
        default="joint_qpos",
        help=(
            "joint_qpos skips unused Curobo MotionGen warmup while retaining MPLib/TOPP qpos execution; "
            "normal initializes Curobo path planning and is only needed for action_type='ee'."
        ),
    )
    parser.add_argument(
        "--save-observation-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save model-input RGB observations once per sampled action chunk as compressed NPZ arrays.",
    )
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally encode cam_high rollout videos. Numeric state/action logs are always saved.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--deviation-plot-degrees",
        type=float,
        default=2.0,
        help="Absolute perturbation magnitude pooled across +/- signs in the actual-state recovery figure.",
    )
    parser.add_argument(
        "--state-plot-max-steps",
        type=int,
        default=100,
        help="Maximum post-perturbation control steps shown in the actual-state recovery figure.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip raw rollouts already represented by metrics.csv.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate caption 70k checkpoints, pi05 20W checkpoint, source HDF5 files, and support bank without loading a model or SAPIEN.",
    )
    parser.add_argument(
        "--environment-preflight-only",
        action="store_true",
        help="Create and replay one expert scene, validate scene identity, then exit before loading a model.",
    )
    args = parser.parse_args()
    args.progresses = _parse_csv(args.progresses, cast=float, name="--progresses")
    args.perturb_degrees = _parse_csv(args.perturb_degrees, cast=float, name="--perturb-degrees")
    if args.episode_source_indices is not None:
        args.episode_source_indices = _parse_csv(
            args.episode_source_indices,
            cast=int,
            name="--episode-source-indices",
        )
    if args.episode_count < 1 or args.rollout_repeats < 1:
        parser.error("--episode-count and --rollout-repeats must be >= 1")
    if any(progress <= 0.0 or progress >= 1.0 for progress in args.progresses):
        parser.error("--progresses values must be within (0, 1)")
    if args.policy_chunk_size < 1 or args.policy_chunk_size > ACTION_HORIZON:
        parser.error(f"--policy-chunk-size must be within [1, {ACTION_HORIZON}]")
    if args.max_policy_actions is not None and args.max_policy_actions < 1:
        parser.error("--max-policy-actions must be positive")
    if args.state_plot_max_steps < 2:
        parser.error("--state-plot-max-steps must be >= 2")
    if args.replay_atol <= 0:
        parser.error("--replay-atol must be positive")
    return args


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _write_json(path: Path, data: Any) -> None:
    def convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported JSON value: {type(value)!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_episode_records(path: Path) -> tuple[EpisodeRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"episode_origin.jsonl not found: {path}")
    records: list[EpisodeRecord] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_name") != TASK_NAME or record.get("task_config") != TASK_CONFIG:
                continue
            records.append(
                EpisodeRecord(
                    global_episode_index=int(record["global_episode_index"]),
                    local_episode_index=int(record["local_episode_index"]),
                    source_episode_index=int(record["source_episode_index"]),
                    episode_length=int(record["episode_length"]),
                    hdf5_path=Path(record["hdf5_path"]),
                    environment_seed=-1,
                )
            )
    records.sort(key=lambda item: item.source_episode_index)
    if len(records) != 50 or [item.source_episode_index for item in records] != list(range(50)):
        raise ValueError("Expected clean place_fan source episodes with source ids exactly 0..49")
    return tuple(records)


def attach_source_environment_seeds(records: tuple[EpisodeRecord, ...]) -> tuple[EpisodeRecord, ...]:
    """Attach the original simulator seed for each successful clean episode."""

    if len(records) != len(PLACE_FAN_SOURCE_SEEDS):
        raise ValueError(
            f"Expected {len(PLACE_FAN_SOURCE_SEEDS)} place_fan clean episodes for the recovered seed map, "
            f"got {len(records)}"
        )
    by_source = {record.source_episode_index: record for record in records}
    if sorted(by_source) != list(range(len(PLACE_FAN_SOURCE_SEEDS))):
        raise ValueError("Source episode ids do not match the recovered place_fan seed map")
    return tuple(
        replace(by_source[source_index], environment_seed=PLACE_FAN_SOURCE_SEEDS[source_index])
        for source_index in sorted(by_source)
    )


def select_records(args: argparse.Namespace, records: tuple[EpisodeRecord, ...]) -> tuple[EpisodeRecord, ...]:
    if args.episode_source_indices is None:
        return records[: args.episode_count]
    by_id = {record.source_episode_index: record for record in records}
    missing = [index for index in args.episode_source_indices if index not in by_id]
    if missing:
        raise ValueError(f"Requested source episode ids outside 0..49: {missing}")
    return tuple(by_id[index] for index in args.episode_source_indices)


def model_specs(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if args.caption_step != CAPTION_STEP:
        raise ValueError(f"caption must be compared at exactly {CAPTION_STEP} steps, got {args.caption_step}")
    if args.pi05_step != PI05_STEP:
        raise ValueError(f"pi05 base must be compared at exactly {PI05_STEP} steps, got {args.pi05_step}")
    return (
        ModelSpec(
            key="caption_support",
            train_config_name=args.caption_config,
            exp_name=args.caption_exp,
            checkpoint_root=args.caption_checkpoint_root.expanduser().resolve(),
            checkpoint_step=CAPTION_STEP,
            use_support_context=True,
            mask_support_video=False,
        ),
        ModelSpec(
            key="caption_masked",
            train_config_name=args.caption_config,
            exp_name=args.caption_exp,
            checkpoint_root=args.caption_checkpoint_root.expanduser().resolve(),
            checkpoint_step=CAPTION_STEP,
            use_support_context=True,
            mask_support_video=True,
        ),
        ModelSpec(
            key="pi05_base",
            train_config_name=args.pi05_config,
            exp_name=args.pi05_exp,
            checkpoint_root=args.pi05_checkpoint_root.expanduser().resolve(),
            checkpoint_step=PI05_STEP,
            use_support_context=False,
            mask_support_video=False,
        ),
    )


def validate_inputs(args: argparse.Namespace, specs: Iterable[ModelSpec], records: Iterable[EpisodeRecord]) -> None:
    support_dir = args.support_bank_root / "human" / TASK_NAME / TASK_CONFIG / args.support_id / args.support_view
    for required in (support_dir / "frames.npy", support_dir / "meta.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Fixed active ego support input is missing: {required}")
    for spec in specs:
        checkpoint = spec.checkpoint_dir
        if spec.key == "pi05_base" and checkpoint.name != "200000":
            raise ValueError(f"{spec.key} must use exactly 200000, got {checkpoint}")
        if spec.key != "pi05_base" and checkpoint.name != "70000":
            raise ValueError(f"{spec.key} must use exactly 70000, got {checkpoint}")
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(
                f"{spec.key} checkpoint missing: {checkpoint / 'params'}. "
                "Do not substitute a different checkpoint step."
            )
        norm_path = checkpoint / "assets" / args.norm_asset_id / "norm_stats.json"
        if not norm_path.is_file():
            raise FileNotFoundError(f"{spec.key} normalization stats missing: {norm_path}")
    for record in records:
        if not record.hdf5_path.is_file():
            raise FileNotFoundError(f"Source trajectory missing: {record.hdf5_path}")
        with h5py.File(record.hdf5_path, "r") as file:
            expected_shape = (record.episode_length, ACTION_DIM)
            if "action" not in file or "observations/qpos" not in file:
                raise KeyError(f"{record.hdf5_path} requires action and observations/qpos")
            if file["action"].shape != expected_shape or file["observations/qpos"].shape != expected_shape:
                raise ValueError(
                    f"{record.hdf5_path} unexpected shapes: action={file['action'].shape}, "
                    f"qpos={file['observations/qpos'].shape}, expected={expected_shape}"
                )


def load_expert(record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(record.hdf5_path, "r") as file:
        return (
            np.asarray(file["action"], dtype=np.float32),
            np.asarray(file["observations/qpos"], dtype=np.float32),
        )


def load_env_args() -> dict[str, Any]:
    task_cfg = yaml.safe_load((ROBOTWIN_ROOT / "task_config" / f"{TASK_CONFIG}.yml").read_text(encoding="utf-8"))
    embodiment_cfg = yaml.safe_load((ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml").read_text(encoding="utf-8"))
    camera_cfg = yaml.safe_load((ROBOTWIN_ROOT / "task_config" / "_camera_config.yml").read_text(encoding="utf-8"))
    embodiment = task_cfg["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"Expected dual-arm single embodiment, got {embodiment}")
    robot_dir = (ROBOTWIN_ROOT / embodiment_cfg[embodiment[0]]["file_path"]).resolve()
    robot_config = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    task_cfg = dict(task_cfg)
    task_cfg.update(
        {
            "task_name": TASK_NAME,
            "task_config": TASK_CONFIG,
            "left_robot_file": str(robot_dir),
            "right_robot_file": str(robot_dir),
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "dual_arm_embodied": True,
            "head_camera_h": camera_cfg[task_cfg["camera"]["head_camera_type"]]["h"],
            "head_camera_w": camera_cfg[task_cfg["camera"]["head_camera_type"]]["w"],
            "control_mode": "joint",
            "eval_mode": True,
            "eval_video_log": False,
            "render_freq": 0,
            "save_data": False,
            "save_path": str(CAPTION_ROOT / "tmp" / "perturbation_recovery"),
        }
    )
    return task_cfg


def configure_warp_compatibility() -> None:
    """Expose the one legacy Warp API used by the vendored Curobo code.

    RoboTwin's Curobo snapshot calls ``wp.torch.device_from_torch`` in exactly
    one place.  Warp 1.16 moved that function to ``wp.device_from_torch`` and
    removed the ``warp.torch`` namespace.  Keep the installed Warp/NumPy
    versions unchanged and provide the old spelling only for this process.
    """

    import warp as wp

    if hasattr(wp, "torch"):
        return
    device_from_torch = getattr(wp, "device_from_torch", None)
    if device_from_torch is None:
        raise RuntimeError(
            f"Warp {getattr(wp, '__version__', 'unknown')} exposes neither "
            "wp.torch.device_from_torch nor wp.device_from_torch"
        )
    wp.torch = SimpleNamespace(device_from_torch=device_from_torch)
    print(
        f"[Warp] compatibility enabled for Warp {getattr(wp, '__version__', 'unknown')}: "
        "wp.torch.device_from_torch -> wp.device_from_torch"
    )


def make_env(env_args: dict[str, Any], record: EpisodeRecord, planner_mode: str):
    configure_warp_compatibility()
    env_module = importlib.import_module(f"envs.{TASK_NAME}")
    configure_recovery_planners(planner_mode)
    if _SHARED_RECOVERY_ENV["env"] is None:
        env = getattr(env_module, TASK_NAME)()
        # Match RoboTwin collection/eval lifecycle: initialize planners once
        # in a disposable bootstrap scene. Every actual source scene below is
        # then reseeded and reset on the same task object without rebuilding
        # planners, so planner initialization cannot shift actor randomness.
        env.setup_demo(
            now_ep_num=record.source_episode_index,
            seed=record.environment_seed,
            is_test=True,
            **env_args,
        )
        close_env(env)
        _SHARED_RECOVERY_ENV["env"] = env
        print("[Environment] planner bootstrap complete; reusing one TASK_ENV for all paired rollouts")
    env = _SHARED_RECOVERY_ENV["env"]
    env.setup_demo(
        now_ep_num=record.source_episode_index,
        seed=record.environment_seed,
        is_test=True,
        **env_args,
    )
    if env.step_lim != 400:
        raise ValueError(f"Expected place_fan eval step limit 400, got {env.step_lim}")
    return env


def close_env(env) -> None:
    try:
        env.close_env(clear_cache=False)
    except Exception:
        with suppress(Exception):
            env.close()


def ensure_robotwin_working_directory() -> None:
    """RoboTwin environment modules resolve asset paths relative to repo root."""

    os.chdir(ROBOTWIN_ROOT)


def configure_recovery_planners(planner_mode: str) -> None:
    """Use RoboTwin's original single-process planner path with fixed YAML paths."""

    robot_module = importlib.import_module("envs.robot.robot")
    planner_module = importlib.import_module("envs.robot.planner")
    robot_class = robot_module.Robot
    configured_mode = getattr(robot_class, "_recovery_planner_mode", None)
    if configured_mode is not None:
        if configured_mode != planner_mode:
            raise RuntimeError(
                f"Recovery planner already configured as {configured_mode!r}, cannot switch to {planner_mode!r}"
            )
        return
    if not hasattr(planner_module, "MotionGen"):
        raise RuntimeError("Curobo MotionGen is unavailable in the evaluation Python environment")
    if planner_mode not in {"normal", "joint_qpos"}:
        raise ValueError(f"Unsupported recovery planner mode: {planner_mode}")
    original_set_planner = robot_class.set_planner

    def source_curobo_config(robot, arm: str) -> Path:
        source = Path(getattr(robot, f"{arm}_curobo_yml_path"))
        candidate = Path(str(source).replace("curobo.yml", f"curobo_{arm}.yml"))
        if candidate.is_file():
            source = candidate
        if not source.is_file():
            raise FileNotFoundError(f"{arm} Curobo config not found: {source}")
        return source

    def fixed_curobo_base_path(robot) -> str:
        output_dir = CAPTION_ROOT / "tmp" / "perturbation_recovery_curobo"
        output_dir.mkdir(parents=True, exist_ok=True)
        doubled_root = str(ROBOTWIN_ROOT / ROBOTWIN_ROOT.name)
        for arm in ("left", "right"):
            if arm == "right" and not robot.has_right_arm:
                continue
            source = source_curobo_config(robot, arm)
            text = source.read_text(encoding="utf-8")
            corrected = text.replace(doubled_root, str(ROBOTWIN_ROOT))
            output = output_dir / f"curobo_{arm}.yml"
            if not output.is_file() or output.read_text(encoding="utf-8") != corrected:
                output.write_text(corrected, encoding="utf-8")
        # Robot.set_planner checks equality before replacing this shared base
        # with curobo_left.yml / curobo_right.yml. Keeping both base paths
        # identical preserves its original non-multiprocessing branch.
        return str(output_dir / "curobo.yml")

    def set_recovery_planner(robot, scene=None) -> None:
        shared_base = fixed_curobo_base_path(robot)
        robot.left_curobo_yml_path = shared_base
        if robot.has_right_arm:
            robot.right_curobo_yml_path = shared_base
        motion_gen_class = planner_module.MotionGen
        original_warmup = motion_gen_class.warmup
        # This experiment exclusively executes joint-qpos actions. RoboTwin's
        # qpos branch uses MPLib/TOPP; MotionGen is only used by action_type='ee'.
        # Its unconditional constructor warmup is therefore unrelated to the
        # evaluated controller and can be skipped without changing qpos rollout.
        if planner_mode == "joint_qpos":
            motion_gen_class.warmup = lambda self, *args, **kwargs: None
        try:
            original_set_planner(robot, scene)
            if robot.communication_flag:
                raise AssertionError("Recovery evaluation must not use forked CUDA planner processes")
        finally:
            motion_gen_class.warmup = original_warmup

    robot_class.set_planner = set_recovery_planner
    robot_class._recovery_planner_mode = planner_mode  # noqa: SLF001
    if planner_mode == "normal":
        print("[Planner] normal Curobo warmup enabled; fixed YAML paths use the original single-process branch")
    else:
        print(
            "[Planner] joint_qpos mode: unused Curobo MotionGen warmup skipped; "
            "MPLib/TOPP and qpos execution retained"
        )


def load_episode_instructions(path: Path, records: Iterable[EpisodeRecord]) -> dict[int, str]:
    """Load the exact language stored alongside each source expert episode."""

    if not path.is_file():
        raise FileNotFoundError(f"LeRobot episode metadata not found: {path}")
    wanted = {record.global_episode_index for record in records}
    instructions: dict[int, str] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            if episode_index not in wanted:
                continue
            tasks = item.get("tasks", [])
            if len(tasks) != 1 or not str(tasks[0]).strip():
                raise ValueError(f"Expected one instruction for global episode {episode_index}, got {tasks!r}")
            instructions[episode_index] = str(tasks[0])
    missing = sorted(wanted - set(instructions))
    if missing:
        raise ValueError(f"Missing instructions for global episodes: {missing}")
    return {
        record.source_episode_index: instructions[record.global_episode_index]
        for record in records
    }


def get_command_qpos(env) -> np.ndarray:
    return np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), dtype=np.float32)


def get_real_qpos(env) -> np.ndarray:
    return np.asarray(
        env.robot.get_left_arm_real_jointState() + env.robot.get_right_arm_real_jointState(),
        dtype=np.float32,
    )


def get_pose(actor) -> np.ndarray:
    pose = actor.get_pose()
    return np.asarray(np.concatenate((pose.p, pose.q)), dtype=np.float32)


def get_state(env) -> dict[str, np.ndarray]:
    return {
        "command_qpos": get_command_qpos(env),
        "real_qpos": get_real_qpos(env),
        "left_ee_pose": np.asarray(env.robot.get_left_ee_pose(), dtype=np.float32),
        "right_ee_pose": np.asarray(env.robot.get_right_ee_pose(), dtype=np.float32),
        "fan_pose": get_pose(env.fan),
        "pad_pose": get_pose(env.pad),
        "gripper_state": np.asarray(
            [env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()], dtype=np.float32
        ),
    }


@contextmanager
def expert_replay_without_early_success(env):
    """Replay the complete recorded expert path instead of eval early-stop.

    RoboTwin's eval ``take_action`` returns as soon as ``check_success`` first
    becomes true, potentially in the middle of an interpolated qpos command.
    The source HDF5 trajectory was produced by completing the scripted expert
    motion, so early-stop would freeze the reconstructed tail at the wrong
    state. Policy rollouts do not use this context and retain normal success
    termination.
    """

    original_check_success = env.check_success
    original_eval_success = bool(env.eval_success)
    env.eval_success = False
    env.check_success = lambda: False
    try:
        yield
    finally:
        env.check_success = original_check_success
        env.eval_success = original_eval_success


def replay_to_frame(
    env,
    expert_actions: np.ndarray,
    expert_qpos: np.ndarray,
    target_frame: int,
    atol: float,
) -> tuple[float, bool]:
    """Replay actions [:target_frame], yielding qpos[target_frame] by dataset convention."""

    if target_frame <= 0 or target_frame >= len(expert_actions):
        raise ValueError(f"target_frame must lie in 1..{len(expert_actions) - 1}, got {target_frame}")
    with expert_replay_without_early_success(env):
        for action in expert_actions[:target_frame]:
            env.take_action(action)
    command_qpos = get_command_qpos(env)
    max_abs_error = float(np.max(np.abs(command_qpos - expert_qpos[target_frame])))
    return max_abs_error, bool(max_abs_error <= atol)


def preflight_environment_replay(
    args: argparse.Namespace,
    env_args: dict[str, Any],
    record: EpisodeRecord,
    instruction: str,
) -> None:
    """Validate environment setup and expert-state reconstruction before model load."""

    expert_actions, expert_qpos = load_expert(record)
    expert_arm, _, _ = active_arm_and_wrist(expert_actions)
    target_frame, actual_progress = target_frame_for_progress(record.episode_length, args.progresses[0])
    env = make_env(env_args, record, args.planner_mode)
    try:
        scene_arm, scene_color = validate_scene_identity(env, record, expert_actions, instruction)
        max_abs_error, replay_ok = replay_to_frame(
            env,
            expert_actions,
            expert_qpos,
            target_frame,
            args.replay_atol,
        )
    finally:
        close_env(env)
    if args.strict_replay and not replay_ok:
        raise RuntimeError(
            f"Environment preflight replay mismatch for source={record.source_episode_index}, "
            f"frame={target_frame}: max_abs={max_abs_error:.6g} > atol={args.replay_atol:.6g}"
        )
    print(
        f"[EnvironmentPreflight] source={record.source_episode_index} frame={target_frame} "
        f"progress={actual_progress:.4f} arm={expert_arm} color={scene_color} "
        f"replay_max_abs={max_abs_error:.6g} ok={replay_ok}"
    )


def active_arm_and_wrist(expert_actions: np.ndarray) -> tuple[str, int, str]:
    """Infer the executing arm from the expert gripper trajectory."""

    actions = np.asarray(expert_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected expert actions [T, {ACTION_DIM}], got {actions.shape}")
    left_activity = float(np.ptp(actions[:, 6]))
    right_activity = float(np.ptp(actions[:, 13]))
    if max(left_activity, right_activity) < 0.5 or min(left_activity, right_activity) > 0.1:
        raise ValueError(
            f"Cannot identify one active gripper: left range={left_activity:.6g}, "
            f"right range={right_activity:.6g}"
        )
    if right_activity > left_activity:
        return "right", 12, "fr_joint6"
    return "left", 5, "fl_joint6"


def validate_scene_identity(
    env,
    record: EpisodeRecord,
    expert_actions: np.ndarray,
    instruction: str,
) -> tuple[str, str]:
    """Reject a seeded scene that cannot be the source expert episode."""

    expert_arm, _, _ = active_arm_and_wrist(expert_actions)
    scene_arm = "right" if float(env.fan.get_pose().p[0]) > 0.0 else "left"
    scene_color = str(env.color_name)
    if scene_arm != expert_arm:
        raise RuntimeError(
            f"Environment scene does not match expert source={record.source_episode_index}: "
            f"expert active arm={expert_arm}, generated scene arm={scene_arm}"
        )
    if scene_color.lower() not in instruction.lower():
        raise RuntimeError(
            f"Environment scene does not match expert source={record.source_episode_index}: "
            f"generated pad color={scene_color!r}, instruction={instruction!r}"
        )
    return scene_arm, scene_color


def apply_wrist_roll_perturbation(env, degrees: float, wrist_index: int) -> np.ndarray | None:
    if abs(degrees) < 1e-12:
        return None
    action = get_command_qpos(env)
    action[wrist_index] += np.float32(math.radians(degrees))
    env.take_action(action)
    return action


def make_model(spec: ModelSpec, args: argparse.Namespace):
    model_args = {
        "checkpoint_root": str(spec.checkpoint_root),
        "asset_id": args.norm_asset_id,
        "use_support_context": spec.use_support_context,
        "mask_support_video": spec.mask_support_video,
        "support_bank_root": str(args.support_bank_root),
        "support_task_name": TASK_NAME,
        "support_task_config": TASK_CONFIG,
        "support_id": args.support_id,
        "support_view": args.support_view,
        "random_support": False,
        "num_support_frames": 8,
    }
    model = PI0(spec.train_config_name, spec.exp_name, spec.checkpoint_step, args.policy_chunk_size, usr_args=model_args)
    if bool(getattr(model, "use_support_context", False)) != spec.use_support_context:
        raise AssertionError(f"{spec.key}: unexpected use_support_context={model.use_support_context}")
    if spec.use_support_context:
        actual_mask = bool(getattr(model, "mask_support_video", False))
        if actual_mask != spec.mask_support_video:
            raise AssertionError(f"{spec.key}: expected mask={spec.mask_support_video}, got {actual_mask}")
        context_mask = np.asarray(model.support_context["support_image_mask"], dtype=bool)
        if spec.mask_support_video and np.any(context_mask):
            raise AssertionError(f"{spec.key}: masked support unexpectedly has valid frames")
        if not spec.mask_support_video and not np.all(context_mask):
            raise AssertionError(f"{spec.key}: active support unexpectedly has masked frames")
    return model


def reset_model_for_rollout(model, instruction: str, sampling_seed: int) -> None:
    """Reset context and make repeated samples paired across conditions."""

    model.reset_obsrvationwindows()
    model.set_language(instruction)
    if not hasattr(model.policy, "_rng"):
        raise TypeError("Expected a JAX policy with a resettable sampling RNG")
    model.policy._rng = jax.random.key(int(sampling_seed))  # noqa: SLF001


def _append_state(log: dict[str, list[np.ndarray]], state: dict[str, np.ndarray]) -> None:
    for key, value in state.items():
        log.setdefault(key, []).append(np.asarray(value, dtype=np.float32))


def _safe_success(env) -> bool:
    if bool(getattr(env, "eval_success", False)):
        return True
    try:
        return bool(env.check_success())
    except Exception:
        return False


def _rgb_observation(observation: dict) -> dict[str, np.ndarray]:
    cameras = observation["observation"]
    return {
        "cam_high": np.asarray(cameras["head_camera"]["rgb"], dtype=np.uint8),
        "cam_left_wrist": np.asarray(cameras["left_camera"]["rgb"], dtype=np.uint8),
        "cam_right_wrist": np.asarray(cameras["right_camera"]["rgb"], dtype=np.uint8),
    }


class _VideoWriter:
    """Optional browser-compatible H.264 writer backed by ffmpeg."""

    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = int(fps)
        self.process: subprocess.Popen | None = None
        self.frame_shape: tuple[int, int, int] | None = None
        self.disabled = False

    def _start(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_shape = (height, width, 3)

    def write(self, rgb: np.ndarray) -> None:
        if self.disabled:
            return
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"[Video] disabled for {self.path.name}: expected RGB [H,W,3], got {frame.shape}")
            self.disabled = True
            return
        if self.process is None:
            try:
                self._start(frame)
            except Exception as error:
                print(f"[Video] disabled for {self.path.name}: {type(error).__name__}: {error}")
                self.disabled = True
                return
        if frame.shape != self.frame_shape:
            print(
                f"[Video] disabled for {self.path.name}: frame shape changed "
                f"from {self.frame_shape} to {frame.shape}"
            )
            self.close()
            self.disabled = True
            return
        try:
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as error:
            print(f"[Video] encoder failed for {self.path.name}: {type(error).__name__}: {error}")
            self.close()
            self.disabled = True

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            print(f"[Video] ffmpeg exited with code {return_code} for {self.path}: {stderr}")


def nearest_future_state_match(
    actual_qpos: np.ndarray,
    expert_actual_qpos: np.ndarray,
    target_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each rollout state to its closest future expert actual-qpos state."""

    actual = np.asarray(actual_qpos, dtype=np.float32)
    expert = np.asarray(expert_actual_qpos, dtype=np.float32)
    if actual.ndim != 2 or actual.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actual qpos [T, {ACTION_DIM}], got {actual.shape}")
    if expert.ndim != 2 or expert.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected expert actual qpos [T, {ACTION_DIM}], got {expert.shape}")
    future_expert = expert[target_frame:]
    if future_expert.size == 0:
        raise ValueError(f"No expert actual states at/after target frame {target_frame}")
    difference = actual[:, None, :] - future_expert[None, :, :]
    distances = np.linalg.norm(difference, axis=-1)
    local_indices = np.argmin(distances, axis=1)
    matched_indices = local_indices.astype(np.int64) + int(target_frame)
    nearest_distance = distances[np.arange(len(actual)), local_indices].astype(np.float32)
    return nearest_distance, matched_indices


def nearest_future_qpos_distance(qpos: np.ndarray, expert_qpos: np.ndarray, target_frame: int) -> np.ndarray:
    """Auxiliary command-level distance retained for backward-compatible raw logs."""

    return nearest_future_state_match(qpos, expert_qpos, target_frame)[0]


def quaternion_angular_error_degrees(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shortest rotation angle between paired wxyz quaternions, in degrees."""

    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 4:
        raise ValueError(f"Expected paired quaternions [T, 4], got {actual.shape} and {reference.shape}")
    actual_norm = np.linalg.norm(actual, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    if np.any(actual_norm <= 1e-12) or np.any(reference_norm <= 1e-12):
        raise ValueError("Quaternion norm must be non-zero")
    actual = actual / actual_norm
    reference = reference / reference_norm
    cosine = np.clip(np.abs(np.sum(actual * reference, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine)).astype(np.float32)


def actual_state_deviation(
    rollout_states: dict[str, np.ndarray],
    expert_reference: dict[str, np.ndarray],
    target_frame: int,
    active_arm: str,
) -> dict[str, np.ndarray]:
    """Compute physical deviations at one coherent nearest expert-state match."""

    real_qpos_distance, matched_indices = nearest_future_state_match(
        rollout_states["real_qpos"],
        expert_reference["real_qpos"],
        target_frame,
    )
    ee_key = f"{active_arm}_ee_pose"
    rollout_ee = np.asarray(rollout_states[ee_key], dtype=np.float32)
    expert_ee = np.asarray(expert_reference[ee_key], dtype=np.float32)[matched_indices]
    rollout_fan = np.asarray(rollout_states["fan_pose"], dtype=np.float32)
    expert_fan = np.asarray(expert_reference["fan_pose"], dtype=np.float32)[matched_indices]
    return {
        "actual_reference_frame_index": matched_indices.astype(np.int64),
        "actual_real_qpos_nearest_future_expert_distance": real_qpos_distance,
        "actual_active_ee_translation_error_m": np.linalg.norm(
            rollout_ee[:, :3] - expert_ee[:, :3], axis=1
        ).astype(np.float32),
        "actual_active_ee_orientation_error_deg": quaternion_angular_error_degrees(
            rollout_ee[:, 3:7], expert_ee[:, 3:7]
        ),
        "actual_fan_position_error_m": np.linalg.norm(
            rollout_fan[:, :3] - expert_fan[:, :3], axis=1
        ).astype(np.float32),
        "actual_fan_orientation_error_deg": quaternion_angular_error_degrees(
            rollout_fan[:, 3:7], expert_fan[:, 3:7]
        ),
    }


def replay_actual_state_alignment(
    state: dict[str, np.ndarray],
    expert_reference: dict[str, np.ndarray],
    target_frame: int,
) -> dict[str, float]:
    """Verify that replay reconstructed the same physical expert state."""

    reference_real_qpos = np.asarray(expert_reference["real_qpos"][target_frame], dtype=np.float32)
    reference_fan = np.asarray(expert_reference["fan_pose"][target_frame], dtype=np.float32)
    actual_real_qpos = np.asarray(state["real_qpos"], dtype=np.float32)
    actual_fan = np.asarray(state["fan_pose"], dtype=np.float32)
    return {
        "actual_real_qpos_max_abs_error": float(np.max(np.abs(actual_real_qpos - reference_real_qpos))),
        "actual_fan_position_error_m": float(np.linalg.norm(actual_fan[:3] - reference_fan[:3])),
        "actual_fan_orientation_error_deg": float(
            quaternion_angular_error_degrees(actual_fan[None, 3:7], reference_fan[None, 3:7])[0]
        ),
    }


def estimate_relift_and_rotation(command_qpos: np.ndarray, fan_pose: np.ndarray, wrist_index: int) -> dict[str, float | int]:
    """Auxiliary, deliberately conservative markers for later video/state inspection."""

    fan_z = np.asarray(fan_pose[:, 2], dtype=np.float32)
    wrist = np.asarray(command_qpos[:, wrist_index], dtype=np.float32)
    if len(fan_z) < 2:
        return {
            "fan_relift_2cm": 0,
            "fan_relift_max_m": 0.0,
            "wrist_total_variation_deg": 0.0,
            "wrist_net_rotation_deg": 0.0,
        }
    running_min = np.minimum.accumulate(fan_z)
    relift = fan_z - running_min
    return {
        "fan_relift_2cm": int(bool(np.any(relift > 0.02))),
        "fan_relift_max_m": float(np.max(relift)),
        "wrist_total_variation_deg": float(np.degrees(np.sum(np.abs(np.diff(wrist))))),
        "wrist_net_rotation_deg": float(np.degrees(wrist[-1] - wrist[0])),
    }


def target_frame_for_progress(episode_length: int, requested_progress: float) -> tuple[int, float]:
    frame = int(round(float(requested_progress) * max(episode_length - 1, 1)))
    frame = min(max(frame, 1), episode_length - 2)
    return frame, frame / max(episode_length - 1, 1)


EXPERT_STATE_KEYS = (
    "command_qpos",
    "real_qpos",
    "left_ee_pose",
    "right_ee_pose",
    "fan_pose",
    "pad_pose",
    "gripper_state",
)


def expert_reference_path(output_dir: Path, record: EpisodeRecord) -> Path:
    return output_dir / "expert_references" / f"source_{record.source_episode_index:02d}" / "actual_states.npz"


def _load_expert_reference(path: Path, record: EpisodeRecord) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        if int(arrays["schema_version"]) != EXPERT_REFERENCE_SCHEMA_VERSION:
            return None
        if int(arrays["source_episode_index"]) != record.source_episode_index:
            return None
        if int(arrays["episode_length"]) != record.episode_length:
            return None
        reference = {key: np.asarray(arrays[key], dtype=np.float32) for key in EXPERT_STATE_KEYS}
    if any(len(value) != record.episode_length for value in reference.values()):
        return None
    return reference


def load_or_build_expert_reference(
    args: argparse.Namespace,
    env_args: dict[str, Any],
    record: EpisodeRecord,
    instruction: str,
) -> dict[str, np.ndarray]:
    """Record the expert's physical simulator trajectory once per source episode."""

    path = expert_reference_path(args.output_dir, record)
    cached = _load_expert_reference(path, record)
    if cached is not None:
        print(f"[ExpertReference] loaded source={record.source_episode_index} path={path}")
        return cached

    expert_actions, expert_qpos = load_expert(record)
    active_arm, _, _ = active_arm_and_wrist(expert_actions)
    env = make_env(env_args, record, args.planner_mode)
    state_log: dict[str, list[np.ndarray]] = {}
    try:
        _, scene_color = validate_scene_identity(env, record, expert_actions, instruction)
        _append_state(state_log, get_state(env))
        with expert_replay_without_early_success(env):
            for action in expert_actions[:-1]:
                env.take_action(action)
                _append_state(state_log, get_state(env))
    finally:
        close_env(env)
    reference = {key: np.asarray(state_log[key], dtype=np.float32) for key in EXPERT_STATE_KEYS}
    if any(len(value) != record.episode_length for value in reference.values()):
        lengths = {key: len(value) for key, value in reference.items()}
        raise RuntimeError(
            f"Expert actual-state reference length mismatch for source={record.source_episode_index}: {lengths}"
        )
    command_error = np.abs(reference["command_qpos"] - expert_qpos)
    max_abs_error = float(np.max(command_error))
    if args.strict_replay and max_abs_error > args.replay_atol:
        raise RuntimeError(
            f"Expert actual-state reference qpos mismatch for source={record.source_episode_index}: "
            f"max_abs={max_abs_error:.6g} > atol={args.replay_atol:.6g}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **reference,
        schema_version=np.asarray(EXPERT_REFERENCE_SCHEMA_VERSION, dtype=np.int64),
        source_episode_index=np.asarray(record.source_episode_index, dtype=np.int64),
        global_episode_index=np.asarray(record.global_episode_index, dtype=np.int64),
        episode_length=np.asarray(record.episode_length, dtype=np.int64),
        active_arm=np.asarray(active_arm),
        scene_color=np.asarray(scene_color),
        command_qpos_max_abs_error=np.asarray(max_abs_error, dtype=np.float32),
        expert_hdf5=np.asarray(str(record.hdf5_path)),
    )
    print(
        f"[ExpertReference] saved source={record.source_episode_index} arm={active_arm} color={scene_color} "
        f"qpos_max_abs={max_abs_error:.6g} path={path}"
    )
    return reference


def raw_rollout_dir(
    output_dir: Path,
    spec: ModelSpec,
    record: EpisodeRecord,
    target_frame: int,
    perturb_degrees: float,
    repeat_index: int,
) -> Path:
    perturb_tag = f"{perturb_degrees:+.1f}deg".replace("+", "plus").replace("-", "minus").replace(".", "p")
    return (
        output_dir
        / "raw_rollouts"
        / spec.key
        / f"source_{record.source_episode_index:02d}"
        / f"frame_{target_frame:03d}"
        / perturb_tag
        / f"repeat_{repeat_index:02d}"
    )


def run_rollout(
    *,
    args: argparse.Namespace,
    spec: ModelSpec,
    model,
    env_args: dict[str, Any],
    record: EpisodeRecord,
    expert_actions: np.ndarray,
    expert_qpos: np.ndarray,
    expert_reference: dict[str, np.ndarray],
    instruction: str,
    requested_progress: float,
    perturb_degrees: float,
    repeat_index: int,
) -> dict[str, Any]:
    """Run one isolated replay -> perturb -> closed-loop policy trajectory."""

    target_frame, actual_progress = target_frame_for_progress(record.episode_length, requested_progress)
    output_path = raw_rollout_dir(
        args.output_dir,
        spec,
        record,
        target_frame,
        perturb_degrees,
        repeat_index,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    sampling_seed = _stable_seed(
        args.policy_seed,
        record.source_episode_index,
        target_frame,
        perturb_degrees,
        repeat_index,
    )
    env = make_env(env_args, record, args.planner_mode)
    video = _VideoWriter(output_path / "cam_high.mp4", args.video_fps) if args.save_videos else None
    try:
        active_arm, wrist_index, wrist_name = active_arm_and_wrist(expert_actions)
        validate_scene_identity(env, record, expert_actions, instruction)
        replay_max_abs_error, replay_ok = replay_to_frame(
            env,
            expert_actions,
            expert_qpos,
            target_frame,
            args.replay_atol,
        )
        if args.strict_replay and not replay_ok:
            raise RuntimeError(
                f"Replay qpos mismatch for source={record.source_episode_index}, frame={target_frame}: "
                f"max_abs={replay_max_abs_error:.6g} > atol={args.replay_atol:.6g}"
            )

        state_before_perturbation = get_state(env)
        replay_actual_alignment = replay_actual_state_alignment(
            state_before_perturbation,
            expert_reference,
            target_frame,
        )
        if args.strict_replay and (
            replay_actual_alignment["actual_real_qpos_max_abs_error"] > args.replay_atol
            or replay_actual_alignment["actual_fan_position_error_m"] > 1e-3
            or replay_actual_alignment["actual_fan_orientation_error_deg"] > 0.5
        ):
            raise RuntimeError(
                f"Physical expert-state replay mismatch for source={record.source_episode_index}, "
                f"frame={target_frame}: {replay_actual_alignment}"
            )
        perturb_action = apply_wrist_roll_perturbation(env, perturb_degrees, wrist_index)
        initial_state = get_state(env)
        if video is not None:
            initial_obs = env.get_obs()
            video.write(_rgb_observation(initial_obs)["cam_high"])

        reset_model_for_rollout(model, instruction, sampling_seed)
        state_log: dict[str, list[np.ndarray]] = {}
        _append_state(state_log, initial_state)
        executed_actions: list[np.ndarray] = []
        predicted_chunks: list[np.ndarray] = []
        chunk_progresses: list[float] = []
        chunk_start_policy_step: list[int] = []
        observation_indices: list[int] = []
        observations: dict[str, list[np.ndarray]] = {
            "cam_high": [],
            "cam_left_wrist": [],
            "cam_right_wrist": [],
        }

        policy_actions = 0
        policy_chunks = 0
        while env.take_action_cnt < env.step_lim and not _safe_success(env):
            if args.max_policy_actions is not None and policy_actions >= args.max_policy_actions:
                break
            observation = env.get_obs()
            rgb = _rgb_observation(observation)
            if args.save_observation_images:
                for key, value in rgb.items():
                    observations[key].append(value)
                observation_indices.append(policy_actions)
            progress = get_chunk_progress(env)
            input_rgb, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb, input_state, chunk_progress=progress)
            actions = np.asarray(model.get_action(), dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, ACTION_DIM):
                raise ValueError(f"{spec.key}: expected action chunk {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}")
            predicted_chunks.append(actions)
            chunk_progresses.append(progress)
            chunk_start_policy_step.append(policy_actions)
            policy_chunks += 1

            for action in actions[: args.policy_chunk_size]:
                if env.take_action_cnt >= env.step_lim or _safe_success(env):
                    break
                if args.max_policy_actions is not None and policy_actions >= args.max_policy_actions:
                    break
                env.take_action(action)
                executed_actions.append(np.asarray(action, dtype=np.float32))
                policy_actions += 1
                current_state = get_state(env)
                _append_state(state_log, current_state)
                if video is not None:
                    obs_after_action = env.get_obs()
                    video.write(_rgb_observation(obs_after_action)["cam_high"])
                # Normal eval refreshes the stored observation after each action.
                if not _safe_success(env):
                    next_observation = env.get_obs()
                    next_rgb, next_state_value = encode_obs(next_observation)
                    model.update_observation_window(
                        next_rgb,
                        next_state_value,
                        chunk_progress=get_chunk_progress(env),
                    )

        success = _safe_success(env)
        stacked_states = {key: np.asarray(values, dtype=np.float32) for key, values in state_log.items()}
        command_distance = nearest_future_qpos_distance(
            stacked_states["command_qpos"],
            expert_qpos,
            target_frame,
        )
        actual_deviation = actual_state_deviation(
            stacked_states,
            expert_reference,
            target_frame,
            active_arm,
        )
        real_distance = actual_deviation["actual_real_qpos_nearest_future_expert_distance"]
        markers = estimate_relift_and_rotation(
            stacked_states["command_qpos"],
            stacked_states["fan_pose"],
            wrist_index,
        )
        executed_action_array = (
            np.stack(executed_actions, axis=0).astype(np.float32)
            if executed_actions
            else np.zeros((0, ACTION_DIM), dtype=np.float32)
        )
        predicted_chunk_array = (
            np.stack(predicted_chunks, axis=0).astype(np.float32)
            if predicted_chunks
            else np.zeros((0, ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        )
        support_mask = (
            np.asarray(model.support_context["support_image_mask"], dtype=np.bool_)
            if spec.use_support_context
            else np.zeros((0,), dtype=np.bool_)
        )
        np.savez_compressed(
            output_path / "states_actions.npz",
            **stacked_states,
            state_before_perturbation_command_qpos=state_before_perturbation["command_qpos"],
            state_before_perturbation_real_qpos=state_before_perturbation["real_qpos"],
            state_after_perturbation_command_qpos=initial_state["command_qpos"],
            state_after_perturbation_real_qpos=initial_state["real_qpos"],
            executed_actions=executed_action_array,
            predicted_action_chunks=predicted_chunk_array,
            chunk_progress=np.asarray(chunk_progresses, dtype=np.float32),
            chunk_start_policy_action=np.asarray(chunk_start_policy_step, dtype=np.int64),
            qpos_nearest_future_expert_distance=command_distance,
            real_qpos_nearest_future_expert_distance=real_distance,
            **actual_deviation,
            expert_actions=expert_actions,
            expert_qpos=expert_qpos,
            target_frame=np.asarray(target_frame, dtype=np.int64),
            wrist_action_index=np.asarray(wrist_index, dtype=np.int64),
            support_image_mask=support_mask,
        )
        if args.save_observation_images:
            np.savez_compressed(
                output_path / "observations.npz",
                **{key: np.asarray(values, dtype=np.uint8) for key, values in observations.items()},
                policy_action_index=np.asarray(observation_indices, dtype=np.int64),
            )
        rollout_metadata = {
            "condition": spec.key,
            "condition_label": CONDITION_LABELS[spec.key],
            "checkpoint_dir": spec.checkpoint_dir,
            "checkpoint_step": spec.checkpoint_step,
            "source_episode_index": record.source_episode_index,
            "global_episode_index": record.global_episode_index,
            "episode_length": record.episode_length,
            "expert_hdf5": record.hdf5_path,
            "requested_progress": requested_progress,
            "target_frame": target_frame,
            "actual_expert_progress": actual_progress,
            "replay_action_count": target_frame,
            "replay_qpos_max_abs_error": replay_max_abs_error,
            "replay_qpos_within_atol": replay_ok,
            "replay_atol": args.replay_atol,
            "replay_actual_state_alignment": replay_actual_alignment,
            "perturbation_degrees": perturb_degrees,
            "perturbation_radians": math.radians(perturb_degrees),
            "perturb_action": perturb_action,
            "active_arm": active_arm,
            "wrist_action_index": wrist_index,
            "wrist_joint_name": wrist_name,
            "support_mode": "masked" if spec.mask_support_video else ("active" if spec.use_support_context else "none"),
            "support_id": args.support_id if spec.use_support_context else "none",
            "support_view": args.support_view if spec.use_support_context else "none",
            "support_image_mask": support_mask,
            "sampling_seed": sampling_seed,
            "repeat_index": repeat_index,
            "instruction": instruction,
            "policy_action_count": policy_actions,
            "policy_chunk_count": policy_chunks,
            "take_action_count_total": int(env.take_action_cnt),
            "step_limit": int(env.step_lim),
            "success": bool(success),
            "stopped_by_max_policy_actions": bool(
                args.max_policy_actions is not None and policy_actions >= args.max_policy_actions and not success
            ),
            "state_file": output_path / "states_actions.npz",
            "expert_reference_file": expert_reference_path(args.output_dir, record),
            "observation_file": output_path / "observations.npz" if args.save_observation_images else None,
            "video_file": output_path / "cam_high.mp4" if args.save_videos else None,
            "actual_active_ee_orientation_error_deg_initial": float(
                actual_deviation["actual_active_ee_orientation_error_deg"][0]
            ),
            "actual_active_ee_orientation_error_deg_final": float(
                actual_deviation["actual_active_ee_orientation_error_deg"][-1]
            ),
            "actual_active_ee_orientation_error_deg_mean": float(
                np.mean(actual_deviation["actual_active_ee_orientation_error_deg"])
            ),
            "actual_fan_orientation_error_deg_initial": float(
                actual_deviation["actual_fan_orientation_error_deg"][0]
            ),
            "actual_fan_orientation_error_deg_final": float(
                actual_deviation["actual_fan_orientation_error_deg"][-1]
            ),
            "actual_fan_orientation_error_deg_mean": float(
                np.mean(actual_deviation["actual_fan_orientation_error_deg"])
            ),
            **markers,
        }
        _write_json(output_path / "metadata.json", rollout_metadata)
        return {
            "condition": spec.key,
            "source_episode_index": record.source_episode_index,
            "global_episode_index": record.global_episode_index,
            "requested_progress": requested_progress,
            "actual_expert_progress": actual_progress,
            "target_frame": target_frame,
            "perturbation_degrees": perturb_degrees,
            "repeat_index": repeat_index,
            "sampling_seed": sampling_seed,
            "success": int(success),
            "policy_action_count": policy_actions,
            "policy_chunk_count": policy_chunks,
            "take_action_count_total": int(env.take_action_cnt),
            "replay_qpos_max_abs_error": replay_max_abs_error,
            "replay_qpos_within_atol": int(replay_ok),
            "active_arm": active_arm,
            "wrist_action_index": wrist_index,
            "wrist_joint_name": wrist_name,
            "support_mode": rollout_metadata["support_mode"],
            "support_id": rollout_metadata["support_id"],
            "support_view": rollout_metadata["support_view"],
            "actual_active_ee_orientation_error_deg_initial": rollout_metadata[
                "actual_active_ee_orientation_error_deg_initial"
            ],
            "actual_active_ee_orientation_error_deg_final": rollout_metadata[
                "actual_active_ee_orientation_error_deg_final"
            ],
            "actual_active_ee_orientation_error_deg_mean": rollout_metadata[
                "actual_active_ee_orientation_error_deg_mean"
            ],
            "actual_fan_orientation_error_deg_initial": rollout_metadata[
                "actual_fan_orientation_error_deg_initial"
            ],
            "actual_fan_orientation_error_deg_final": rollout_metadata[
                "actual_fan_orientation_error_deg_final"
            ],
            "actual_fan_orientation_error_deg_mean": rollout_metadata[
                "actual_fan_orientation_error_deg_mean"
            ],
            "fan_relift_2cm": markers["fan_relift_2cm"],
            "fan_relift_max_m": markers["fan_relift_max_m"],
            "wrist_total_variation_deg": markers["wrist_total_variation_deg"],
            "wrist_net_rotation_deg": markers["wrist_net_rotation_deg"],
            "raw_rollout_dir": str(output_path),
        }
    finally:
        if video is not None:
            video.close()
        close_env(env)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            parsed: dict[str, Any] = dict(row)
            for key in (
                "source_episode_index",
                "global_episode_index",
                "target_frame",
                "repeat_index",
                "sampling_seed",
                "success",
                "policy_action_count",
                "policy_chunk_count",
                "take_action_count_total",
                "replay_qpos_within_atol",
                "wrist_action_index",
                "fan_relift_2cm",
            ):
                if key in parsed:
                    parsed[key] = int(parsed[key])
            for key in (
                "requested_progress",
                "actual_expert_progress",
                "perturbation_degrees",
                "replay_qpos_max_abs_error",
                "fan_relift_max_m",
                "wrist_total_variation_deg",
                "wrist_net_rotation_deg",
                "actual_active_ee_orientation_error_deg_initial",
                "actual_active_ee_orientation_error_deg_final",
                "actual_active_ee_orientation_error_deg_mean",
                "actual_fan_orientation_error_deg_initial",
                "actual_fan_orientation_error_deg_final",
                "actual_fan_orientation_error_deg_mean",
            ):
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _resume_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {
        (
            row["condition"],
            row["source_episode_index"],
            round(float(row["requested_progress"]), 8),
            round(float(row["perturbation_degrees"]), 8),
            row["repeat_index"],
        )
        for row in rows
    }


def validate_resume_schema(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    required = {
        "actual_active_ee_orientation_error_deg_initial",
        "actual_active_ee_orientation_error_deg_final",
        "actual_active_ee_orientation_error_deg_mean",
        "actual_fan_orientation_error_deg_initial",
        "actual_fan_orientation_error_deg_final",
        "actual_fan_orientation_error_deg_mean",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(
            "Existing metrics.csv predates actual-state recovery metrics. "
            f"Use a new --output-dir instead of mixing schemas; missing={missing}"
        )


def _mean_and_sem(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
    return mean, sem, count


def summarize_metrics(rows: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovery_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            float(row["requested_progress"]),
            float(row["perturbation_degrees"]),
        )
        grouped.setdefault(key, []).append(row)
    for (condition, progress, perturbation), group in sorted(grouped.items()):
        success = np.asarray([row["success"] for row in group], dtype=np.float64)
        success_mean, success_sem, sample_count = _mean_and_sem(success)
        recovery_rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "requested_progress": progress,
                "perturbation_degrees": perturbation,
                "rollout_count": sample_count,
                "recovery_rate": success_mean,
                "recovery_rate_sem": success_sem,
                "success_count": int(np.sum(success)),
            }
        )
        behavior_row = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "requested_progress": progress,
            "perturbation_degrees": perturbation,
            "rollout_count": sample_count,
        }
        for metric in (
            "fan_relift_2cm",
            "fan_relift_max_m",
            "wrist_total_variation_deg",
            "wrist_net_rotation_deg",
            "policy_action_count",
            "replay_qpos_max_abs_error",
            "actual_active_ee_orientation_error_deg_initial",
            "actual_active_ee_orientation_error_deg_final",
            "actual_active_ee_orientation_error_deg_mean",
            "actual_fan_orientation_error_deg_initial",
            "actual_fan_orientation_error_deg_final",
            "actual_fan_orientation_error_deg_mean",
        ):
            mean, sem, _ = _mean_and_sem(np.asarray([row[metric] for row in group], dtype=np.float64))
            behavior_row[f"{metric}_mean"] = mean
            behavior_row[f"{metric}_sem"] = sem
        behavior_rows.append(behavior_row)
    _write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    _write_csv(output_dir / "behavior_summary.csv", behavior_rows)
    return recovery_rows, behavior_rows


def summarize_recovery_magnitude(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Pool positive/negative perturbations at each absolute magnitude."""

    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            float(row["requested_progress"]),
            abs(float(row["perturbation_degrees"])),
        )
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (condition, progress, magnitude), group in sorted(grouped.items()):
        success = np.asarray([row["success"] for row in group], dtype=np.float64)
        mean, sem, count = _mean_and_sem(success)
        summary_rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "requested_progress": progress,
                "perturbation_magnitude_degrees": magnitude,
                "rollout_count": count,
                "recovery_rate": mean,
                "recovery_rate_sem": sem,
                "success_count": int(np.sum(success)),
                "signs_pooled": "0" if magnitude == 0.0 else "-+",
            }
        )
    _write_csv(output_dir / "recovery_magnitude_summary.csv", summary_rows)
    return summary_rows


def plot_recovery_rate(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    progresses = sorted({float(row["requested_progress"]) for row in summary_rows})
    figure, axes = plt.subplots(1, len(progresses), figsize=(5.1 * len(progresses), 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, progress in zip(axes, progresses, strict=True):
        for condition in CONDITION_ORDER:
            rows = [
                row
                for row in summary_rows
                if row["condition"] == condition and float(row["requested_progress"]) == progress
            ]
            rows.sort(key=lambda row: float(row["perturbation_magnitude_degrees"]))
            if not rows:
                continue
            x = np.asarray([row["perturbation_magnitude_degrees"] for row in rows], dtype=np.float64)
            y = np.asarray([row["recovery_rate"] for row in rows], dtype=np.float64)
            yerr = np.asarray([row["recovery_rate_sem"] for row in rows], dtype=np.float64)
            axis.plot(x, y, marker="o", linewidth=2.0, color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition])
            axis.fill_between(x, np.clip(y - yerr, 0.0, 1.0), np.clip(y + yerr, 0.0, 1.0), color=CONDITION_COLORS[condition], alpha=0.16)
        axis.set_title(f"expert progress {progress:.2f}")
        axis.set_xlabel("absolute wrist-roll perturbation (deg)")
        axis.set_xticks([0, 1, 2, 5])
        axis.set_xlim(-0.25, 5.25)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("recovery success rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output_dir / "figures" / "recovery_rate_vs_perturbation.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "figures" / "recovery_rate_vs_perturbation.pdf", bbox_inches="tight")
    plt.close(figure)


def _state_series_for_row(row: dict[str, Any], key: str) -> np.ndarray:
    raw_path = Path(row["raw_rollout_dir"]) / "states_actions.npz"
    with np.load(raw_path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise KeyError(f"{raw_path} is missing required actual-state metric {key!r}")
        return np.asarray(arrays[key], dtype=np.float64)


def _mean_sem_variable_series(series: list[np.ndarray], max_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = min(max(len(values) for values in series), max_steps)
    padded = np.full((len(series), length), np.nan, dtype=np.float64)
    for index, values in enumerate(series):
        used = min(len(values), length)
        padded[index, :used] = values[:used]
    mean = np.nanmean(padded, axis=0)
    count = np.sum(np.isfinite(padded), axis=0)
    sem = np.zeros(length, dtype=np.float64)
    for step in range(length):
        values = padded[:, step]
        values = values[np.isfinite(values)]
        if len(values) > 1:
            sem[step] = np.std(values, ddof=1) / math.sqrt(len(values))
    return mean, sem, count


def plot_actual_state_recovery(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected = [
        row
        for row in rows
        if np.isclose(abs(float(row["perturbation_degrees"])), abs(args.deviation_plot_degrees), atol=1e-8)
    ]
    if not selected:
        print(
            f"[Plot] no rollouts at abs perturbation={abs(args.deviation_plot_degrees):g}; "
            "skipping actual-state recovery figure"
        )
        return
    progresses = sorted({float(row["requested_progress"]) for row in selected})
    metric_rows = (
        (
            "actual_active_ee_orientation_error_deg",
            "active EE orientation error (deg)",
        ),
        (
            "actual_fan_orientation_error_deg",
            "fan orientation error (deg)",
        ),
    )
    figure, axes = plt.subplots(
        len(metric_rows),
        len(progresses),
        figsize=(5.1 * len(progresses), 3.7 * len(metric_rows)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for metric_index, (metric_key, ylabel) in enumerate(metric_rows):
        for progress_index, progress in enumerate(progresses):
            axis = axes[metric_index, progress_index]
            for condition in CONDITION_ORDER:
                series = [
                    _state_series_for_row(row, metric_key)
                    for row in selected
                    if row["condition"] == condition and np.isclose(row["requested_progress"], progress)
                ]
                if not series:
                    continue
                mean, sem, count = _mean_sem_variable_series(series, args.state_plot_max_steps)
                x = np.arange(len(mean))
                valid = count > 0
                axis.plot(
                    x[valid],
                    mean[valid],
                    linewidth=2.0,
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition],
                )
                axis.fill_between(
                    x[valid],
                    np.maximum(mean[valid] - sem[valid], 0.0),
                    mean[valid] + sem[valid],
                    color=CONDITION_COLORS[condition],
                    alpha=0.16,
                )
            if metric_index == 0:
                axis.set_title(f"expert progress {progress:.2f}")
            if metric_index == len(metric_rows) - 1:
                axis.set_xlabel("control steps after perturbation")
            if progress_index == 0:
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        f"Actual-state recovery after ±{abs(args.deviation_plot_degrees):g}° wrist-roll perturbation",
        y=0.97,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(
        args.output_dir / "figures" / "actual_state_recovery_after_perturbation.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "figures" / "actual_state_recovery_after_perturbation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_metrics_npz(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    magnitude_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    np.savez_compressed(
        output_dir / "metrics.npz",
        condition=np.asarray([row["condition"] for row in rows], dtype="<U32"),
        source_episode_index=np.asarray([row["source_episode_index"] for row in rows], dtype=np.int64),
        target_frame=np.asarray([row["target_frame"] for row in rows], dtype=np.int64),
        requested_progress=np.asarray([row["requested_progress"] for row in rows], dtype=np.float32),
        actual_expert_progress=np.asarray([row["actual_expert_progress"] for row in rows], dtype=np.float32),
        perturbation_degrees=np.asarray([row["perturbation_degrees"] for row in rows], dtype=np.float32),
        repeat_index=np.asarray([row["repeat_index"] for row in rows], dtype=np.int64),
        success=np.asarray([row["success"] for row in rows], dtype=np.int8),
        policy_action_count=np.asarray([row["policy_action_count"] for row in rows], dtype=np.int64),
        replay_qpos_max_abs_error=np.asarray([row["replay_qpos_max_abs_error"] for row in rows], dtype=np.float32),
        fan_relift_2cm=np.asarray([row["fan_relift_2cm"] for row in rows], dtype=np.int8),
        fan_relift_max_m=np.asarray([row["fan_relift_max_m"] for row in rows], dtype=np.float32),
        wrist_total_variation_deg=np.asarray([row["wrist_total_variation_deg"] for row in rows], dtype=np.float32),
        wrist_net_rotation_deg=np.asarray([row["wrist_net_rotation_deg"] for row in rows], dtype=np.float32),
        actual_active_ee_orientation_error_deg_initial=np.asarray(
            [row["actual_active_ee_orientation_error_deg_initial"] for row in rows], dtype=np.float32
        ),
        actual_active_ee_orientation_error_deg_final=np.asarray(
            [row["actual_active_ee_orientation_error_deg_final"] for row in rows], dtype=np.float32
        ),
        actual_active_ee_orientation_error_deg_mean=np.asarray(
            [row["actual_active_ee_orientation_error_deg_mean"] for row in rows], dtype=np.float32
        ),
        actual_fan_orientation_error_deg_initial=np.asarray(
            [row["actual_fan_orientation_error_deg_initial"] for row in rows], dtype=np.float32
        ),
        actual_fan_orientation_error_deg_final=np.asarray(
            [row["actual_fan_orientation_error_deg_final"] for row in rows], dtype=np.float32
        ),
        actual_fan_orientation_error_deg_mean=np.asarray(
            [row["actual_fan_orientation_error_deg_mean"] for row in rows], dtype=np.float32
        ),
        summary_condition=np.asarray([row["condition"] for row in summary_rows], dtype="<U32"),
        summary_progress=np.asarray([row["requested_progress"] for row in summary_rows], dtype=np.float32),
        summary_perturbation_degrees=np.asarray([row["perturbation_degrees"] for row in summary_rows], dtype=np.float32),
        summary_recovery_rate=np.asarray([row["recovery_rate"] for row in summary_rows], dtype=np.float32),
        summary_recovery_rate_sem=np.asarray([row["recovery_rate_sem"] for row in summary_rows], dtype=np.float32),
        summary_rollout_count=np.asarray([row["rollout_count"] for row in summary_rows], dtype=np.int64),
        magnitude_summary_condition=np.asarray([row["condition"] for row in magnitude_rows], dtype="<U32"),
        magnitude_summary_progress=np.asarray(
            [row["requested_progress"] for row in magnitude_rows], dtype=np.float32
        ),
        magnitude_summary_degrees=np.asarray(
            [row["perturbation_magnitude_degrees"] for row in magnitude_rows], dtype=np.float32
        ),
        magnitude_summary_recovery_rate=np.asarray(
            [row["recovery_rate"] for row in magnitude_rows], dtype=np.float32
        ),
        magnitude_summary_recovery_rate_sem=np.asarray(
            [row["recovery_rate_sem"] for row in magnitude_rows], dtype=np.float32
        ),
        magnitude_summary_rollout_count=np.asarray(
            [row["rollout_count"] for row in magnitude_rows], dtype=np.int64
        ),
    )


def expected_rollout_count(args: argparse.Namespace, record_count: int, spec_count: int) -> int:
    return record_count * len(args.progresses) * len(args.perturb_degrees) * args.rollout_repeats * spec_count


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.episode_origin = args.episode_origin.expanduser().resolve()
    args.support_bank_root = args.support_bank_root.expanduser().resolve()
    ensure_robotwin_working_directory()
    all_records = attach_source_environment_seeds(load_episode_records(args.episode_origin))
    records = select_records(args, all_records)
    specs = model_specs(args)
    validate_inputs(args, specs, records)
    run_metadata = {
        "task_name": TASK_NAME,
        "task_config": TASK_CONFIG,
        "caption_step": CAPTION_STEP,
        "pi05_step": PI05_STEP,
        "model_specs": [asdict(spec) | {"checkpoint_dir": spec.checkpoint_dir} for spec in specs],
        "episode_records": [asdict(record) for record in records],
        "requested_progresses": args.progresses,
        "perturb_degrees": args.perturb_degrees,
        "rollout_repeats": args.rollout_repeats,
        "expected_rollout_count": expected_rollout_count(args, len(records), len(specs)),
        "support_input": {
            "support_id": args.support_id,
            "support_view": args.support_view,
            "support_bank_root": args.support_bank_root,
            "caption_support_mode": "active",
            "caption_masked_mode": "null frames with support_image_mask=false",
        },
        "replay": {
            "method": (
                "one shared RoboTwin TASK_ENV/planner bootstrap; reseed and reset the scene, "
                "then expert action prefix replay"
            ),
            "alignment_check": "max abs command qpos against HDF5 observations/qpos[target_frame]",
            "atol": args.replay_atol,
        },
        "instruction_source": args.episode_origin.with_name("episodes.jsonl"),
        "planner_mode": args.planner_mode,
        "planner_validity": (
            "formal joint-qpos expert-state evaluation; MPLib/TOPP execution retained"
            if args.planner_mode == "joint_qpos"
            else "formal evaluation with Curobo MotionGen initialized"
        ),
        "perturbation": {
            "type": "absolute qpos command offset",
            "joint_choice": "fan side: left fl_joint6 action[5], right fr_joint6 action[12]",
            "joint_semantics": "sixth arm joint / wrist roll; URDF local revolute axis [1, 0, 0]",
        },
        "actual_state_recovery": {
            "reference": "physical expert simulator trajectory recorded from the matched source scene",
            "temporal_match": "nearest future expert state by actual robot qpos",
            "primary_metrics": [
                "active end-effector quaternion angular error in degrees",
                "fan quaternion angular error in degrees",
            ],
            "auxiliary_metrics": [
                "actual robot qpos distance",
                "active end-effector translation error",
                "fan position error",
                "command qpos distance",
            ],
        },
    }
    _write_json(args.output_dir / "run_metadata.json", run_metadata)
    print(
        f"[Plan] selected clean episodes={len(records)} progress={args.progresses} perturb={args.perturb_degrees} "
        f"conditions={list(CONDITION_ORDER)} expected_rollouts={run_metadata['expected_rollout_count']}"
    )
    for spec in specs:
        print(f"[Checkpoint] {spec.key}: {spec.checkpoint_dir}")
    if args.dry_run:
        print("[DryRun] inputs validated; no policy, SAPIEN environment, or rollout was started.")
        return

    env_args = load_env_args()
    instruction_metadata = args.episode_origin.with_name("episodes.jsonl")
    instructions = load_episode_instructions(instruction_metadata, records)
    _write_json(args.output_dir / "instructions.json", instructions)
    for preflight_record in records:
        preflight_environment_replay(
            args,
            env_args,
            preflight_record,
            instructions[preflight_record.source_episode_index],
        )
    if args.environment_preflight_only:
        print("[EnvironmentPreflightOnly] validation passed; no model was loaded and no rollout was started.")
        return
    expert_references = {
        record.source_episode_index: load_or_build_expert_reference(
            args,
            env_args,
            record,
            instructions[record.source_episode_index],
        )
        for record in records
    }
    metrics_path = args.output_dir / "metrics.csv"
    all_rows = _read_metrics(metrics_path) if args.resume else []
    validate_resume_schema(all_rows)
    completed = _resume_keys(all_rows)

    for spec in specs:
        print(f"[LoadModel] {spec.key} (step={spec.checkpoint_step})")
        model = make_model(spec, args)
        for record in records:
            expert_actions, expert_qpos = load_expert(record)
            for requested_progress in args.progresses:
                target_frame, _ = target_frame_for_progress(record.episode_length, requested_progress)
                for perturb_degrees in args.perturb_degrees:
                    for repeat_index in range(args.rollout_repeats):
                        key = (
                            spec.key,
                            record.source_episode_index,
                            round(float(requested_progress), 8),
                            round(float(perturb_degrees), 8),
                            repeat_index,
                        )
                        if key in completed:
                            print(
                                f"[Resume] {spec.key} source={record.source_episode_index} "
                                f"frame={target_frame} perturb={perturb_degrees:+g} repeat={repeat_index}"
                            )
                            continue
                        print(
                            f"[Rollout] {spec.key} source={record.source_episode_index} frame={target_frame} "
                            f"progress={requested_progress:.3f} perturb={perturb_degrees:+g}deg repeat={repeat_index}"
                        )
                        row = run_rollout(
                            args=args,
                            spec=spec,
                            model=model,
                            env_args=env_args,
                            record=record,
                            expert_actions=expert_actions,
                            expert_qpos=expert_qpos,
                            expert_reference=expert_references[record.source_episode_index],
                            instruction=instructions[record.source_episode_index],
                            requested_progress=requested_progress,
                            perturb_degrees=perturb_degrees,
                            repeat_index=repeat_index,
                        )
                        all_rows.append(row)
                        completed.add(key)
                        _write_csv(metrics_path, all_rows)
        del model
        gc.collect()

    all_rows = _read_metrics(metrics_path)
    summary_rows, behavior_rows = summarize_metrics(all_rows, args.output_dir)
    magnitude_rows = summarize_recovery_magnitude(all_rows, args.output_dir)
    save_metrics_npz(all_rows, summary_rows, magnitude_rows, args.output_dir)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_recovery_rate(magnitude_rows, args.output_dir)
    plot_actual_state_recovery(all_rows, args)
    main_figure_paths = [
        figure_dir / "recovery_rate_vs_perturbation.png",
        figure_dir / "actual_state_recovery_after_perturbation.png",
    ]
    _write_json(
        args.output_dir / "summary.json",
        {
            "completed_rollouts": len(all_rows),
            "expected_rollouts": run_metadata["expected_rollout_count"],
            "metrics_csv": metrics_path,
            "metrics_npz": args.output_dir / "metrics.npz",
            "recovery_summary": args.output_dir / "recovery_summary.csv",
            "recovery_magnitude_summary": args.output_dir / "recovery_magnitude_summary.csv",
            "behavior_summary": args.output_dir / "behavior_summary.csv",
            "figures": [path for path in main_figure_paths if path.is_file()],
            "notes": {
                "actual_state_reference": (
                    "real qpos selects one nearest future expert frame; EE/fan pose errors use that same frame"
                ),
                "command_qpos_reference": "saved only as an auxiliary raw metric and not plotted as a main figure",
                "relift_rotation_markers": "auxiliary heuristic signals, not task-success definitions",
            },
        },
    )
    print(f"[Done] completed={len(all_rows)} results={args.output_dir}")


if __name__ == "__main__":
    main()
