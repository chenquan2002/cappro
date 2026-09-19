"""Validate the local CapPro evaluation chain without loading model weights."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path

from flax import nnx
from flax import traverse_util
import jax
import numpy as np

from openpi.policies.support_video import SupportVideoBank
from openpi.training import config as training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--support-bank-root", required=True, type=Path)
    parser.add_argument("--support-view", required=True)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()
    robotwin_root = args.robotwin_root.expanduser().resolve()
    os.chdir(robotwin_root)
    policy_root = Path(__file__).resolve().parents[1]
    checkpoint = policy_root / "checkpoints" / args.train_config_name / args.model_name / args.checkpoint_id

    require_file(robotwin_root / "script" / "eval_policy.py", "RoboTwin evaluator")
    require_file(robotwin_root / "task_config" / f"{args.task_config}.yml", "RoboTwin task config")
    require_file(robotwin_root / "envs" / f"{args.task_name}.py", "RoboTwin task environment")
    require_file(checkpoint / "_CHECKPOINT_METADATA", "checkpoint metadata")
    require_file(checkpoint / "params" / "_METADATA", "checkpoint parameter metadata")

    task_module = importlib.import_module(f"envs.{args.task_name}")
    task_class = getattr(task_module, args.task_name, None)
    if not isinstance(task_class, type):
        raise AttributeError(f"envs.{args.task_name} does not export task class {args.task_name}")

    module = importlib.import_module(args.policy_name)
    expected_policy_module = (policy_root / "__init__.py").resolve()
    if Path(module.__file__).resolve() != expected_policy_module:
        raise ImportError(
            f"Imported {args.policy_name} from {module.__file__}, expected {expected_policy_module}"
        )
    for function_name in ("get_model", "eval", "reset_model"):
        if not callable(getattr(module, function_name, None)):
            raise AttributeError(f"{args.policy_name} does not export callable {function_name}")
    pi0_class = getattr(module, "PI0", None)
    if not isinstance(pi0_class, type):
        raise AttributeError(f"{args.policy_name} does not export PI0")
    pi_model_module = importlib.import_module(pi0_class.__module__)
    expected_pi_model_module = (policy_root / "pi_model.py").resolve()
    if Path(pi_model_module.__file__).resolve() != expected_pi_model_module:
        raise ImportError(
            f"Imported PI0 from {pi_model_module.__file__}, expected {expected_pi_model_module}"
        )

    config = training_config.get_config(args.train_config_name)
    if not bool(getattr(config.model, "use_support_context", False)):
        raise ValueError(f"Training config does not enable support context: {args.train_config_name}")
    num_support_frames = int(getattr(config.model, "num_support_frames", 0))
    if num_support_frames < 1:
        raise ValueError(f"Invalid num_support_frames={num_support_frames}")

    model = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
    model_keys = set(traverse_util.flatten_dict(nnx.state(model).to_pure_dict()))
    parameter_metadata = json.loads((checkpoint / "params" / "_METADATA").read_text(encoding="utf-8"))
    checkpoint_keys = {
        tuple(ast.literal_eval(serialized_key)[1:-1])
        for serialized_key in parameter_metadata["tree_metadata"]
    }
    if model_keys != checkpoint_keys:
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        raise ValueError(
            "Checkpoint parameter tree does not match the current model: "
            f"missing={missing}, unexpected={unexpected}"
        )

    asset_dirs = sorted(path for path in (checkpoint / "assets").iterdir() if path.is_dir())
    if len(asset_dirs) != 1:
        raise ValueError(f"Expected exactly one checkpoint asset directory, got: {asset_dirs}")
    norm_stats_path = asset_dirs[0] / "norm_stats.json"
    require_file(norm_stats_path, "checkpoint norm stats")
    norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8")).get("norm_stats", {})
    for key in ("state", "actions"):
        mean = np.asarray(norm_stats.get(key, {}).get("mean", []))
        if mean.shape != (14,):
            raise ValueError(f"Expected 14-D {key} norm stats, got shape {mean.shape}: {norm_stats_path}")

    support_bank = SupportVideoBank(args.support_bank_root, num_frames=num_support_frames)
    candidates = support_bank.discover(args.task_name, args.task_config, view=args.support_view)
    for support_id, view in candidates:
        support_bank.load(args.task_name, args.task_config, support_id, view)

    print("[Eval preflight] OK")
    print(f"  policy module: {module.__file__}")
    print(f"  PI0 module: {pi_model_module.__file__}")
    print(f"  task module: {task_module.__file__}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  norm stats: {norm_stats_path}")
    print(f"  parameter leaves: {len(checkpoint_keys)}")
    print(f"  support bank: {args.support_bank_root.expanduser().resolve()}")
    print(f"  support candidates ({args.support_view}): {len(candidates)}")


if __name__ == "__main__":
    main()
