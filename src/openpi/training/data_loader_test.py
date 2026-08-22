import dataclasses
import json

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _IndexedDataset:
    episode_data_index = {
        "from": [0, 2, 5],
        "to": [2, 5, 6],
    }

    def __getitem__(self, index):
        return int(index)

    def __len__(self):
        return 6


def test_episode_subset_maps_selected_episodes():
    subset = _data_loader.EpisodeSubset(_IndexedDataset(), [2, 0])

    assert len(subset) == 3
    assert [subset[i] for i in range(len(subset))] == [5, 0, 1]
    assert subset[-1] == 1


def test_source_episode_indices(tmp_path):
    origin_path = tmp_path / "episode_origin.jsonl"
    records = [
        {"global_episode_index": index, "task_name": task_name}
        for index, task_name in enumerate(sorted(_data_loader._SOURCE_TASKS))
    ]
    records.append({"global_episode_index": 99, "task_name": "other_task"})
    origin_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert _data_loader._source_episode_indices(origin_path) == list(range(13))


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
