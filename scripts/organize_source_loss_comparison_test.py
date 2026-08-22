import csv
import json

from scripts import organize_source_loss_comparison as organize


def _write_run(root, model, mode, clean_loss, randomized_loss):
    run_dir = root / "_runs" / model / mode
    run_dir.mkdir(parents=True)
    rows = [
        {
            "task_name": "click_alarmclock",
            "task_config": "all",
            "action_loss": (clean_loss + 10 * randomized_loss) / 11,
            "action_loss_std": 0.1,
            "sample_count": 550,
        },
        {
            "task_name": "click_alarmclock",
            "task_config": "demo_clean",
            "action_loss": clean_loss,
            "action_loss_std": 0.1,
            "sample_count": 50,
        },
        {
            "task_name": "click_alarmclock",
            "task_config": "demo_randomized",
            "action_loss": randomized_loss,
            "action_loss_std": 0.1,
            "sample_count": 500,
        },
    ]
    with (run_dir / "taskwise.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_kind": model,
                "config_name": model,
                "exp_name": "exp",
                "step": "30000",
                "selection_file": "selection.npz",
                "selection_sha256": "selection-hash",
                "norm_stats_sha256": "norm-hash",
                "batch_size": 8,
                "rng_steps": [0],
                "loss_seed": 12345,
            }
        )
    )


def _write_task_run(root, model, mode, task_name, clean_loss, randomized_loss):
    run_dir = root / "_runs" / model / mode / task_name
    run_dir.mkdir(parents=True)
    rows = [
        {
            "task_name": task_name,
            "task_config": "all",
            "action_loss": (clean_loss + randomized_loss) / 2,
            "action_loss_std": 0.1,
            "sample_count": 20,
        },
        {
            "task_name": task_name,
            "task_config": "demo_clean",
            "action_loss": clean_loss,
            "action_loss_std": 0.1,
            "sample_count": 10,
        },
        {
            "task_name": task_name,
            "task_config": "demo_randomized",
            "action_loss": randomized_loss,
            "action_loss_std": 0.1,
            "sample_count": 10,
        },
    ]
    with (run_dir / "taskwise.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_kind": model,
                "config_name": model,
                "exp_name": "exp",
                "step": "30000",
                "selection_file": f"{task_name}.npz",
                "selection_sha256": f"{task_name}-selection-hash",
                "input_probe_sha256": f"{task_name}-input-hash",
                "norm_stats_sha256": "norm-hash",
                "batch_size": 8,
                "action_horizon": 50,
                "action_dim": 14,
                "rng_steps": [0],
                "loss_seed": 12345,
                "preprocess_mode": "eval",
                "repo_id": "source_data_hovapi_repo",
            }
        )
    )


def test_organize_builds_balanced_per_task_comparison(tmp_path):
    _write_run(tmp_path, "support_caption", "uniform", 1.0, 3.0)
    _write_run(tmp_path, "pi05_3w", "uniform", 2.0, 4.0)

    organize.organize(tmp_path)

    support = json.loads(
        (tmp_path / "click_alarmclock/support_caption/uniform/summary.json").read_text()
    )
    comparison = json.loads((tmp_path / "click_alarmclock/comparison.json").read_text())
    assert support["balanced"]["action_loss"] == 2.0
    assert comparison["modes"]["uniform"]["balanced"] == {
        "support_caption_action_loss": 2.0,
        "pi05_3w_action_loss": 3.0,
        "delta_support_minus_pi05": -1.0,
        "relative_delta": -1 / 3,
    }


def test_organize_reads_full_task_runs(tmp_path):
    _write_task_run(tmp_path, "support_caption", "full", "click_alarmclock", 1.0, 3.0)
    _write_task_run(tmp_path, "pi05_3w", "full", "click_alarmclock", 2.0, 4.0)

    organize.organize(tmp_path)

    support = json.loads(
        (tmp_path / "click_alarmclock/support_caption/full/summary.json").read_text()
    )
    metadata = json.loads(
        (tmp_path / "click_alarmclock/support_caption/full/metadata.json").read_text()
    )
    comparison = json.loads((tmp_path / "click_alarmclock/comparison.json").read_text())
    assert support["balanced"]["action_loss"] == 2.0
    assert metadata["source_run"].endswith("_runs/support_caption/full/click_alarmclock")
    assert comparison["modes"]["full"]["balanced"]["delta_support_minus_pi05"] == -1.0
