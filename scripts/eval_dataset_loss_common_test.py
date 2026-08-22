import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts import eval_dataset_loss_common as common


class _TinyEpisodeDataset:
    def __init__(self):
        self.episode_data_index = {
            "from": np.asarray([0, 3, 7], dtype=np.int64),
            "to": np.asarray([3, 7, 9], dtype=np.int64),
        }

    def __len__(self):
        return 9

    def __getitem__(self, index):
        return {"index": np.asarray(index, dtype=np.int64)}


def test_running_loss_stats_uses_token_weighted_caption_loss():
    stats = common.RunningLossStats()
    stats.update(
        action_loss=1.0,
        caption_loss=2.0,
        caption_accuracy=0.5,
        caption_token_count=2,
        hand_loss=4.0,
        hand_accuracy=0.0,
        hand_token_count=1,
        support_valid=True,
    )
    stats.update(
        action_loss=3.0,
        caption_loss=4.0,
        caption_accuracy=1.0,
        caption_token_count=6,
        hand_loss=2.0,
        hand_accuracy=1.0,
        hand_token_count=3,
        support_valid=False,
    )

    result = stats.finalize(caption_loss_weight=0.1)

    assert result["action_loss"] == 2.0
    assert result["caption_loss"] == 3.5
    assert result["caption_token_accuracy"] == 0.875
    assert result["caption_hand_side_loss"] == 2.5
    assert result["joint_loss"] == 2.35
    assert result["support_valid_samples"] == 1


def test_running_loss_stats_action_only_omits_caption_fields():
    stats = common.RunningLossStats()
    stats.update(
        action_loss=0.25,
        caption_loss=0.0,
        caption_accuracy=0.0,
        caption_token_count=0,
        hand_loss=0.0,
        hand_accuracy=0.0,
        hand_token_count=0,
        support_valid=False,
    )

    result = stats.finalize(caption_loss_weight=0.1, include_caption=False)

    assert result == {
        "action_loss": 0.25,
        "action_loss_std": 0.0,
        "sample_count": 1,
    }


def test_running_loss_stats_reports_sampled_action_mse_when_present():
    stats = common.RunningLossStats()
    stats.update(
        action_loss=1.0,
        caption_loss=0.0,
        caption_accuracy=0.0,
        caption_token_count=0,
        hand_loss=0.0,
        hand_accuracy=0.0,
        hand_token_count=0,
        support_valid=False,
        sampled_action_mse=2.0,
        sampled_action_mse_t0=4.0,
    )
    stats.update(
        action_loss=3.0,
        caption_loss=0.0,
        caption_accuracy=0.0,
        caption_token_count=0,
        hand_loss=0.0,
        hand_accuracy=0.0,
        hand_token_count=0,
        support_valid=False,
        sampled_action_mse=6.0,
        sampled_action_mse_t0=8.0,
    )

    result = stats.finalize(caption_loss_weight=0.1, include_caption=False)

    assert result["sampled_action_mse"] == 4.0
    assert result["sampled_action_mse_t0"] == 6.0
    assert result["sampled_action_mse_count"] == 2


def test_build_first_demo_manifest_selects_lowest_demo(tmp_path):
    records = [
        {
            "global_episode_index": 7,
            "support_round_id": 0,
            "support_type": "human",
            "support_id": "human_demo_003",
            "support_frames_npy": "/bank/human_demo_003/front/frames.npy",
        },
        {
            "global_episode_index": 7,
            "support_round_id": 8,
            "support_type": "human",
            "support_id": "human_demo_000",
            "support_frames_npy": "/bank/human_demo_000/right/frames.npy",
        },
    ]

    output_path, chosen = common.build_first_demo_manifest(records, [7], tmp_path / "first.jsonl")

    assert chosen[7]["support_id"] == "human_demo_000"
    assert chosen[7]["support_round_id"] == 0
    assert json.loads(output_path.read_text())["support_id"] == "human_demo_000"


def test_uniform_and_full_selection_use_global_dataset_indices():
    dataset = _TinyEpisodeDataset()

    uniform = common.build_uniform_selection(dataset, [1], samples_per_episode=3)
    full = common.build_full_selection(dataset, [1])

    np.testing.assert_array_equal(uniform.base_indices, np.asarray([3, 5, 6]))
    np.testing.assert_array_equal(uniform.frame_indices, np.asarray([0, 2, 3]))
    np.testing.assert_array_equal(full.base_indices, np.asarray([3, 4, 5, 6]))
    np.testing.assert_array_equal(full.episode_indices, np.asarray([1, 1, 1, 1]))


def test_shared_selection_round_trip_and_validates_dataset_mapping(tmp_path):
    dataset = _TinyEpisodeDataset()
    selection = common.build_uniform_selection(dataset, [1], samples_per_episode=3)
    context = {"version": 1, "mode": "uniform", "task_name": "task"}
    path = tmp_path / "selection.npz"

    saved_digest = common.save_shared_selection(path, selection, context)
    loaded, loaded_digest = common.load_shared_selection(path, context)
    common.validate_selection(loaded, dataset, [1])

    assert saved_digest == loaded_digest
    np.testing.assert_array_equal(loaded.base_indices, selection.base_indices)
    np.testing.assert_array_equal(loaded.frame_indices, selection.frame_indices)
    assert loaded.train_positions is None


def test_shared_selection_rejects_different_context(tmp_path):
    selection = common.build_full_selection(_TinyEpisodeDataset(), [1])
    path = tmp_path / "selection.npz"
    common.save_shared_selection(path, selection, {"mode": "full"})

    with np.testing.assert_raises_regex(ValueError, "context mismatch"):
        common.load_shared_selection(path, {"mode": "uniform"})


def test_validate_selection_rejects_wrong_base_index():
    selection = common.EvalSelection(
        base_indices=np.asarray([4]),
        episode_indices=np.asarray([1]),
        frame_indices=np.asarray([0]),
        support_round_ids=np.asarray([0]),
    )

    with np.testing.assert_raises_regex(ValueError, "base index mismatch"):
        common.validate_selection(selection, _TinyEpisodeDataset(), [1])


def test_training_permutation_matches_torch_dataloader():
    size = 20
    seed = 42
    loader = DataLoader(
        list(range(size)),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    loader_order = np.concatenate([batch.numpy() for batch in loader])

    replay_order = common._training_permutation_prefix(size, size, seed=seed)  # noqa: SLF001

    np.testing.assert_array_equal(replay_order, loader_order)


def test_trainlike_selection_filters_task_episode_after_global_shuffle(tmp_path):
    selection = common.build_or_load_trainlike_selection(
        _TinyEpisodeDataset(),
        training_episode_ids=[0, 1, 2],
        selected_episode_ids=[1],
        replay_file=tmp_path / "replay.npz",
        train_steps=5,
        train_batch_size=4,
        support_rounds_per_cycle=2,
        eval_samples=6,
        replay_seed=3,
        sample_method="linspace",
        sample_seed=0,
        force_remake=False,
    )

    assert len(selection.base_indices) <= 6
    assert set(selection.episode_indices.tolist()) == {1}
    assert np.all((selection.base_indices >= 3) & (selection.base_indices < 7))
    assert np.all((selection.support_round_ids >= 0) & (selection.support_round_ids < 2))


def test_target_task_is_selectable_from_all_scope():
    episode_info = {
        1: common.EpisodeInfo(1, "click_alarmclock", "demo_clean", 4),
        2: common.EpisodeInfo(2, "turn_switch", "demo_clean", 5),
    }
    records = [
        {
            "global_episode_index": 2,
            "task_name": "turn_switch",
            "support_type": "human",
            "has_support": True,
        }
    ]

    selected = common.select_episode_ids(
        episode_info,
        [1, 2],
        records,
        task_names=("turn_switch",),
        task_config="all",
        data_scope="all",
    )

    assert selected == (2,)
