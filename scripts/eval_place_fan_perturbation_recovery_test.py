from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.eval_place_fan_perturbation_recovery as recovery


def test_quaternion_angular_error_is_sign_invariant() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    same_rotation = -identity
    np.testing.assert_allclose(
        recovery.quaternion_angular_error_degrees(identity, same_rotation),
        [0.0],
        atol=1e-6,
    )


def test_quaternion_angular_error_reports_degrees() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rotate_z_90 = np.asarray([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=np.float32)
    np.testing.assert_allclose(
        recovery.quaternion_angular_error_degrees(identity, rotate_z_90),
        [90.0],
        atol=1e-4,
    )


def test_nearest_future_state_match_returns_global_reference_indices() -> None:
    expert = np.zeros((4, recovery.ACTION_DIM), dtype=np.float32)
    expert[:, 0] = [0.0, 1.0, 2.0, 3.0]
    actual = np.zeros((2, recovery.ACTION_DIM), dtype=np.float32)
    actual[:, 0] = [1.1, 2.9]
    distance, indices = recovery.nearest_future_state_match(actual, expert, target_frame=1)
    np.testing.assert_allclose(distance, [0.1, 0.1], atol=1e-6)
    np.testing.assert_array_equal(indices, [1, 3])


def test_actual_state_deviation_uses_one_coherent_reference_match() -> None:
    expert_qpos = np.zeros((3, recovery.ACTION_DIM), dtype=np.float32)
    expert_qpos[:, 0] = [0.0, 1.0, 2.0]
    rollout_qpos = np.zeros((2, recovery.ACTION_DIM), dtype=np.float32)
    rollout_qpos[:, 0] = [1.1, 1.9]
    identity_pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    expert_pose = np.repeat(identity_pose[None], 3, axis=0)
    rollout_pose = np.repeat(identity_pose[None], 2, axis=0)
    rollout_states = {
        "real_qpos": rollout_qpos,
        "right_ee_pose": rollout_pose,
        "fan_pose": rollout_pose,
    }
    reference = {
        "real_qpos": expert_qpos,
        "right_ee_pose": expert_pose,
        "fan_pose": expert_pose,
    }
    result = recovery.actual_state_deviation(rollout_states, reference, target_frame=1, active_arm="right")
    np.testing.assert_array_equal(result["actual_reference_frame_index"], [1, 2])
    np.testing.assert_allclose(result["actual_active_ee_orientation_error_deg"], 0.0, atol=1e-6)
    np.testing.assert_allclose(result["actual_fan_orientation_error_deg"], 0.0, atol=1e-6)


def test_recovery_magnitude_summary_pools_positive_and_negative(tmp_path) -> None:
    rows = [
        {"condition": "caption_support", "requested_progress": 0.65, "perturbation_degrees": 0.0, "success": 1},
        {"condition": "caption_support", "requested_progress": 0.65, "perturbation_degrees": -2.0, "success": 0},
        {"condition": "caption_support", "requested_progress": 0.65, "perturbation_degrees": 2.0, "success": 1},
    ]
    summary = recovery.summarize_recovery_magnitude(rows, tmp_path)
    by_magnitude = {row["perturbation_magnitude_degrees"]: row for row in summary}
    assert by_magnitude[0.0]["rollout_count"] == 1
    assert by_magnitude[0.0]["recovery_rate"] == 1.0
    assert by_magnitude[2.0]["rollout_count"] == 2
    assert by_magnitude[2.0]["recovery_rate"] == 0.5


def test_actual_state_plot_reads_physical_metrics(tmp_path) -> None:
    rows = []
    for condition in recovery.CONDITION_ORDER:
        rollout_dir = tmp_path / "raw" / condition
        rollout_dir.mkdir(parents=True)
        np.savez_compressed(
            rollout_dir / "states_actions.npz",
            actual_active_ee_orientation_error_deg=np.linspace(2.0, 0.5, 8, dtype=np.float32),
            actual_fan_orientation_error_deg=np.linspace(4.0, 1.0, 8, dtype=np.float32),
        )
        rows.append(
            {
                "condition": condition,
                "requested_progress": 0.65,
                "perturbation_degrees": 2.0,
                "raw_rollout_dir": str(rollout_dir),
            }
        )
    args = SimpleNamespace(output_dir=tmp_path, deviation_plot_degrees=2.0, state_plot_max_steps=100)
    (tmp_path / "figures").mkdir()
    recovery.plot_actual_state_recovery(rows, args)
    assert (tmp_path / "figures" / "actual_state_recovery_after_perturbation.png").is_file()
    assert not (tmp_path / "figures" / "state_deviation_after_perturbation.png").exists()
