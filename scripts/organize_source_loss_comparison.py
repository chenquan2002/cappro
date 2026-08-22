#!/usr/bin/env python3
"""Create per-task views and paired action-loss tables from shared source evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MODELS = ("support_caption", "pi05_3w")
MODES = ("uniform", "full", "trainlike")
BALANCED_METRICS = (
    "action_loss",
    "sampled_action_mse",
    "sampled_action_mse_t0",
    "caption_loss",
    "caption_token_accuracy",
    "caption_hand_side_loss",
    "caption_hand_side_accuracy",
    "caption_weighted_loss",
    "joint_loss",
)
PAIRED_METADATA_FIELDS = (
    "selection_sha256",
    "input_probe_sha256",
    "norm_stats_sha256",
    "batch_size",
    "action_horizon",
    "action_dim",
    "rng_steps",
    "loss_seed",
    "sample_action_mse",
    "sample_action_mse_only",
    "sample_action_num_steps",
    "preprocess_mode",
    "repo_id",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def parse_value(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value else number


def load_task_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            task_name = row.pop("task_name")
            task_config = row.pop("task_config")
            result.setdefault(task_name, {})[task_config] = {
                key: parse_value(value) for key, value in row.items()
            }
    return result


def balanced_metrics(configs: dict[str, dict[str, Any]]) -> dict[str, float]:
    clean = configs.get("demo_clean", {})
    randomized = configs.get("demo_randomized", {})
    return {
        metric: (float(clean[metric]) + float(randomized[metric])) / 2
        for metric in BALANCED_METRICS
        if metric in clean and metric in randomized
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_task_bundle(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    taskwise_path = run_dir / "taskwise.csv"
    if not metadata_path.is_file() or not taskwise_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_rows = load_task_rows(taskwise_path)
    return {
        "run_dir": run_dir,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None,
        "metadata": metadata,
        "task_rows": task_rows,
    }


def load_runs_for_model_mode(root: Path, model_name: str, mode: str) -> dict[str, list[dict[str, Any]]]:
    run_dir = root / "_runs" / model_name / mode
    task_runs: dict[str, list[dict[str, Any]]] = {}

    direct_bundle = load_task_bundle(run_dir)
    if direct_bundle is not None and mode != "full":
        for task_name, configs in direct_bundle["task_rows"].items():
            task_runs.setdefault(task_name, []).append(
                {
                    "run_dir": run_dir,
                    "configs": configs,
                    "metadata": direct_bundle["metadata"],
                }
            )
        return task_runs

    if mode != "full" or not run_dir.is_dir():
        return task_runs

    for task_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        loaded_chunks = 0
        for chunk_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            chunk_bundle = load_task_bundle(chunk_dir)
            if chunk_bundle is None:
                continue
            for task_name, configs in chunk_bundle["task_rows"].items():
                task_runs.setdefault(task_name, []).append(
                    {
                        "run_dir": chunk_dir,
                        "configs": configs,
                        "metadata": chunk_bundle["metadata"],
                    }
                )
                loaded_chunks += 1
        if loaded_chunks:
            continue
        direct_task_bundle = load_task_bundle(task_dir)
        if direct_task_bundle is not None:
            for task_name, configs in direct_task_bundle["task_rows"].items():
                task_runs.setdefault(task_name, []).append(
                    {
                        "run_dir": task_dir,
                        "configs": configs,
                        "metadata": direct_task_bundle["metadata"],
                    }
                )
    return task_runs


def combine_metric_dicts(
    metric_dicts: list[dict[str, dict[str, Any]]],
    *,
    caption_loss_weight: float,
) -> dict[str, dict[str, Any]]:
    class Accumulator:
        def __init__(self) -> None:
            self.action_sum = 0.0
            self.action_sumsq = 0.0
            self.action_count = 0
            self.sample_count = 0
            self.sampled_action_mse_sum = 0.0
            self.sampled_action_mse_sumsq = 0.0
            self.sampled_action_mse_t0_sum = 0.0
            self.sampled_action_mse_t0_sumsq = 0.0
            self.sampled_action_mse_count = 0
            self.caption_nll_sum = 0.0
            self.caption_correct_sum = 0.0
            self.caption_token_count = 0
            self.hand_nll_sum = 0.0
            self.hand_correct_sum = 0.0
            self.hand_token_count = 0
            self.caption_valid_samples = 0
            self.support_valid_samples = 0
            self.include_caption = False

        def update(self, metrics: dict[str, Any]) -> None:
            count = int(metrics["sample_count"])
            self.sample_count += count
            if "action_loss" in metrics:
                action_mean = float(metrics["action_loss"])
                action_std = float(metrics.get("action_loss_std", 0.0))
                self.action_sum += action_mean * count
                self.action_sumsq += (action_std**2 + action_mean**2) * count
                self.action_count += count
            if "sampled_action_mse" in metrics:
                mse_count = int(metrics.get("sampled_action_mse_count", count))
                sampled_mean = float(metrics["sampled_action_mse"])
                sampled_std = float(metrics.get("sampled_action_mse_std", 0.0))
                sampled_t0_mean = float(metrics.get("sampled_action_mse_t0", sampled_mean))
                sampled_t0_std = float(metrics.get("sampled_action_mse_t0_std", 0.0))
                self.sampled_action_mse_sum += sampled_mean * mse_count
                self.sampled_action_mse_sumsq += (sampled_std**2 + sampled_mean**2) * mse_count
                self.sampled_action_mse_t0_sum += sampled_t0_mean * mse_count
                self.sampled_action_mse_t0_sumsq += (sampled_t0_std**2 + sampled_t0_mean**2) * mse_count
                self.sampled_action_mse_count += mse_count
            if "caption_loss" not in metrics:
                return
            self.include_caption = True
            caption_tokens = int(metrics.get("caption_token_count", 0))
            hand_tokens = int(metrics.get("caption_hand_side_token_count", 0))
            self.caption_nll_sum += float(metrics["caption_loss"]) * caption_tokens
            self.caption_correct_sum += float(metrics["caption_token_accuracy"]) * caption_tokens
            self.caption_token_count += caption_tokens
            self.hand_nll_sum += float(metrics["caption_hand_side_loss"]) * hand_tokens
            self.hand_correct_sum += float(metrics["caption_hand_side_accuracy"]) * hand_tokens
            self.hand_token_count += hand_tokens
            self.caption_valid_samples += int(metrics.get("caption_valid_samples", 0))
            self.support_valid_samples += int(metrics.get("support_valid_samples", 0))

        def finalize(self) -> dict[str, Any]:
            result: dict[str, Any] = {
                "sample_count": self.sample_count,
            }
            if self.action_count:
                action_denominator = max(self.action_count, 1)
                action_mean = self.action_sum / action_denominator
                action_variance = max(self.action_sumsq / action_denominator - action_mean**2, 0.0)
                result.update(
                    {
                        "action_loss": action_mean,
                        "action_loss_std": action_variance**0.5,
                    }
                )
            if self.sampled_action_mse_count:
                mse_denominator = max(self.sampled_action_mse_count, 1)
                sampled_mean = self.sampled_action_mse_sum / mse_denominator
                sampled_variance = max(
                    self.sampled_action_mse_sumsq / mse_denominator - sampled_mean**2,
                    0.0,
                )
                sampled_t0_mean = self.sampled_action_mse_t0_sum / mse_denominator
                sampled_t0_variance = max(
                    self.sampled_action_mse_t0_sumsq / mse_denominator - sampled_t0_mean**2,
                    0.0,
                )
                result.update(
                    {
                        "sampled_action_mse": sampled_mean,
                        "sampled_action_mse_std": sampled_variance**0.5,
                        "sampled_action_mse_t0": sampled_t0_mean,
                        "sampled_action_mse_t0_std": sampled_t0_variance**0.5,
                        "sampled_action_mse_count": self.sampled_action_mse_count,
                    }
                )
            if not self.include_caption:
                return result
            caption_denominator = max(self.caption_token_count, 1)
            hand_denominator = max(self.hand_token_count, 1)
            caption_loss = self.caption_nll_sum / caption_denominator
            caption_weighted_loss = caption_loss_weight * caption_loss
            result.update(
                {
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
                }
            )
            return result

    accumulators: dict[str, Accumulator] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            accumulators.setdefault(key, Accumulator()).update(value)
    return {key: acc.finalize() for key, acc in accumulators.items()}


def merge_task_runs(
    task_runs: list[dict[str, Any]],
    *,
    task_name: str,
    model_name: str,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not task_runs:
        raise ValueError("task_runs must not be empty")
    task_runs = sorted(task_runs, key=lambda item: str(item["run_dir"]))
    first_metadata = task_runs[0]["metadata"]
    caption_loss_weight = float(first_metadata.get("caption_loss_weight", 0.1))
    for bundle in task_runs[1:]:
        metadata = bundle["metadata"]
        for field in (
            "model_kind",
            "config_name",
            "exp_name",
            "step",
            "norm_asset_id",
            "norm_stats_sha256",
            "batch_size",
            "action_horizon",
            "action_dim",
            "rng_steps",
            "loss_seed",
            "sample_action_mse",
            "sample_action_mse_only",
            "sample_action_num_steps",
            "preprocess_mode",
            "repo_id",
            "task_name",
            "task_config",
            "data_scope",
            "support_view",
            "caption_max_len",
            "caption_loss_weight",
        ):
            if first_metadata.get(field) != metadata.get(field):
                raise ValueError(
                    f"Chunk metadata mismatch for {model_name}/{mode}/{task_name}: "
                    f"{field}={first_metadata.get(field)!r} vs {metadata.get(field)!r}"
                )

    combined_metrics = combine_metric_dicts(
        [bundle["configs"] for bundle in task_runs],
        caption_loss_weight=caption_loss_weight,
    )
    summary = {
        "task_name": task_name,
        "model": model_name,
        "mode": mode,
        "metrics": combined_metrics,
        "balanced": balanced_metrics(combined_metrics),
    }
    first_run_dir = Path(task_runs[0]["run_dir"])
    source_run = first_run_dir.parent if mode == "full" and first_run_dir.name.startswith("chunk_") else first_run_dir
    metadata = {
        "source_run": str(source_run),
        "source_runs": [str(bundle["run_dir"]) for bundle in task_runs],
        "chunk_count": len(task_runs),
        "model_kind": first_metadata["model_kind"],
        "config_name": first_metadata["config_name"],
        "exp_name": first_metadata["exp_name"],
        "step": first_metadata["step"],
        "selection_file": first_metadata.get("selection_file"),
        "selection_sha256s": [bundle["metadata"].get("selection_sha256") for bundle in task_runs],
        "input_probe_sha256s": [bundle["metadata"].get("input_probe_sha256") for bundle in task_runs],
        "norm_stats_sha256": first_metadata["norm_stats_sha256"],
        "batch_size": first_metadata["batch_size"],
        "action_horizon": first_metadata.get("action_horizon"),
        "action_dim": first_metadata.get("action_dim"),
        "rng_steps": first_metadata["rng_steps"],
        "loss_seed": first_metadata["loss_seed"],
        "sample_action_mse": bool(first_metadata.get("sample_action_mse", False)),
        "sample_action_mse_only": bool(first_metadata.get("sample_action_mse_only", False)),
        "sample_action_num_steps": first_metadata.get("sample_action_num_steps"),
        "caption_loss_weight": caption_loss_weight,
        "task_name": first_metadata.get("task_name", task_name),
        "task_config": first_metadata.get("task_config", "all"),
        "data_scope": first_metadata.get("data_scope"),
        "support_view": first_metadata.get("support_view"),
    }
    return summary, metadata


def chunk_signature(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metadata.get("selection_sha256"),
        metadata.get("input_probe_sha256"),
        metadata.get("norm_stats_sha256"),
        metadata.get("batch_size"),
        metadata.get("action_horizon"),
        metadata.get("action_dim"),
        tuple(metadata.get("rng_steps", [])),
        metadata.get("loss_seed"),
        metadata.get("sample_action_mse", False),
        metadata.get("sample_action_mse_only", False),
        metadata.get("sample_action_num_steps"),
        metadata.get("preprocess_mode"),
        metadata.get("repo_id"),
        metadata.get("task_name"),
        metadata.get("task_config"),
        metadata.get("data_scope"),
        metadata.get("support_view"),
        metadata.get("episode_chunk_size"),
        metadata.get("episode_chunk_index"),
        metadata.get("evaluated_samples"),
        metadata.get("evaluated_episodes"),
    )


def organize(root: Path) -> None:
    runs: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    task_names: set[str] = set()
    for model_name in MODELS:
        for mode in MODES:
            loaded = load_runs_for_model_mode(root, model_name, mode)
            if not loaded:
                continue
            runs[(model_name, mode)] = loaded
            task_names.update(loaded)

    for mode in MODES:
        support_key = ("support_caption", mode)
        base_key = ("pi05_3w", mode)
        if support_key not in runs or base_key not in runs:
            continue
        paired_tasks = sorted(set(runs[support_key]) & set(runs[base_key]))
        for task_name in paired_tasks:
            support_runs = runs[support_key][task_name]
            base_runs = runs[base_key][task_name]
            if len(support_runs) != len(base_runs):
                raise ValueError(
                    f"Paired {mode}/{task_name} chunk count mismatch: "
                    f"{len(support_runs)} vs {len(base_runs)}"
                )
            if mode == "full":
                support_chunks = [chunk_signature(bundle["metadata"]) for bundle in support_runs]
                base_chunks = [chunk_signature(bundle["metadata"]) for bundle in base_runs]
                if support_chunks != base_chunks:
                    raise ValueError(
                        f"Paired {mode}/{task_name} chunk metadata mismatch: "
                        f"{support_chunks} vs {base_chunks}"
                    )
            else:
                support_metadata = support_runs[0]["metadata"]
                base_metadata = base_runs[0]["metadata"]
                mismatches = {
                    field: (support_metadata.get(field), base_metadata.get(field))
                    for field in PAIRED_METADATA_FIELDS
                    if support_metadata.get(field) != base_metadata.get(field)
                }
                if mismatches:
                    raise ValueError(
                        f"Paired {mode}/{task_name} evaluation metadata mismatch: {mismatches}"
                    )

    comparison_rows: list[dict[str, Any]] = []
    for task_name in sorted(task_names):
        task_comparison: dict[str, Any] = {"task_name": task_name, "modes": {}}
        for model_name in MODELS:
            for mode in MODES:
                key = (model_name, mode)
                if task_name not in runs.get(key, {}):
                    continue
                summary, metadata = merge_task_runs(
                    runs[key][task_name],
                    task_name=task_name,
                    model_name=model_name,
                    mode=mode,
                )
                output_dir = root / task_name / model_name / mode
                write_json(output_dir / "summary.json", summary)
                write_json(output_dir / "metadata.json", metadata)

        for mode in MODES:
            support = runs.get(("support_caption", mode), {}).get(task_name)
            base = runs.get(("pi05_3w", mode), {}).get(task_name)
            if support is None or base is None:
                continue
            support_summary, _ = merge_task_runs(support, task_name=task_name, model_name="support_caption", mode=mode)
            base_summary, _ = merge_task_runs(base, task_name=task_name, model_name="pi05_3w", mode=mode)
            mode_comparison: dict[str, Any] = {}
            support_configs = {**support_summary["metrics"], "balanced": support_summary["balanced"]}
            base_configs = {**base_summary["metrics"], "balanced": base_summary["balanced"]}
            for task_config in ("all", "demo_clean", "demo_randomized", "balanced"):
                if task_config not in support_configs or task_config not in base_configs:
                    continue
                has_flow_loss = (
                    "action_loss" in support_configs[task_config]
                    and "action_loss" in base_configs[task_config]
                )
                has_sampled_mse = (
                    "sampled_action_mse" in support_configs[task_config]
                    and "sampled_action_mse" in base_configs[task_config]
                )
                if not has_flow_loss and not has_sampled_mse:
                    continue
                mode_comparison[task_config] = {}
                if has_flow_loss:
                    support_loss = float(support_configs[task_config]["action_loss"])
                    base_loss = float(base_configs[task_config]["action_loss"])
                    delta = support_loss - base_loss
                    relative = delta / base_loss if base_loss != 0 else None
                    mode_comparison[task_config].update(
                        {
                            "support_caption_action_loss": support_loss,
                            "pi05_3w_action_loss": base_loss,
                            "delta_support_minus_pi05": delta,
                            "relative_delta": relative,
                        }
                    )
                if has_sampled_mse:
                    support_mse = float(support_configs[task_config]["sampled_action_mse"])
                    base_mse = float(base_configs[task_config]["sampled_action_mse"])
                    mse_delta = support_mse - base_mse
                    mse_relative = mse_delta / base_mse if base_mse != 0 else None
                    support_mse_t0 = float(support_configs[task_config].get("sampled_action_mse_t0", support_mse))
                    base_mse_t0 = float(base_configs[task_config].get("sampled_action_mse_t0", base_mse))
                    mse_t0_delta = support_mse_t0 - base_mse_t0
                    mse_t0_relative = mse_t0_delta / base_mse_t0 if base_mse_t0 != 0 else None
                    mode_comparison[task_config].update(
                        {
                            "support_caption_sampled_action_mse": support_mse,
                            "pi05_3w_sampled_action_mse": base_mse,
                            "sampled_action_mse_delta_support_minus_pi05": mse_delta,
                            "sampled_action_mse_relative_delta": mse_relative,
                            "support_caption_sampled_action_mse_t0": support_mse_t0,
                            "pi05_3w_sampled_action_mse_t0": base_mse_t0,
                            "sampled_action_mse_t0_delta_support_minus_pi05": mse_t0_delta,
                            "sampled_action_mse_t0_relative_delta": mse_t0_relative,
                        }
                    )
                comparison_rows.append(
                    {
                        "task_name": task_name,
                        "mode": mode,
                        "task_config": task_config,
                        **mode_comparison[task_config],
                    }
                )
            task_comparison["modes"][mode] = mode_comparison
        write_json(root / task_name / "comparison.json", task_comparison)

    if comparison_rows:
        fieldnames = list(comparison_rows[0])
        for row in comparison_rows[1:]:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
    write_json(
        root / "comparison_metadata.json",
        {
            "models": list(MODELS),
            "modes": list(MODES),
            "tasks": sorted(task_names),
            "available_runs": [list(key) for key in sorted(runs)],
        },
    )
    print(f"[Organize] tasks={len(task_names)} runs={len(runs)} root={root}")


if __name__ == "__main__":
    organize(parse_args().root.expanduser().resolve())
