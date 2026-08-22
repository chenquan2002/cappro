import bisect
from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.support_context as _support_context
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

_SOURCE_TASKS = {
    "click_alarmclock",
    "press_stapler",
    "open_laptop",
    "rotate_qrcode",
    "handover_mic",
    "pick_dual_bottles",
    "grab_roller",
    "move_stapler_pad",
    "place_empty_cup",
    "place_bread_basket",
    "place_a2b_left",
    "stack_blocks_two",
    "place_fan",
}


def _parse_csv_env(name: str) -> set[str] | None:
    """Read an optional comma-separated environment variable as a string set."""
    value = os.getenv(name, "").strip()
    if not value:
        return None
    values = {item.strip() for item in value.split(",") if item.strip()}
    if not values:
        return None
    return values


def _selected_episode_indices(
    episode_origin_path: str | Path,
    *,
    source_only: bool,
    task_names: set[str] | None,
    task_configs: set[str] | None,
) -> list[int]:
    """Return global episode indices matching the requested training subset."""
    episode_origin_path = Path(episode_origin_path)
    if not episode_origin_path.is_file():
        raise FileNotFoundError(f"Episode origin manifest not found: {episode_origin_path}")

    found_tasks: set[str] = set()
    found_selected_tasks: set[str] = set()
    found_configs: set[str] = set()
    episode_indices: list[int] = []
    with episode_origin_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task_name = str(record["task_name"])
            task_config = str(record["task_config"])
            if source_only and task_name not in _SOURCE_TASKS:
                continue
            if task_names is not None and task_name not in task_names:
                continue
            if task_configs is not None and task_config not in task_configs:
                continue

            if source_only:
                found_tasks.add(task_name)
            found_selected_tasks.add(task_name)
            found_configs.add(task_config)
            episode_indices.append(int(record["global_episode_index"]))

    required_tasks = _SOURCE_TASKS if task_names is None else (_SOURCE_TASKS & task_names)
    missing_tasks = sorted(required_tasks - found_tasks) if source_only else []
    if missing_tasks:
        raise ValueError(f"Source tasks are absent from {episode_origin_path}: {missing_tasks}")
    if task_names is not None:
        missing_task_filters = sorted(task_names - found_selected_tasks)
        if missing_task_filters:
            raise ValueError(
                f"Requested training task(s) are absent after filtering {episode_origin_path}: "
                f"{missing_task_filters}"
            )
    if task_configs is not None:
        missing_config_filters = sorted(task_configs - found_configs)
        if missing_config_filters:
            raise ValueError(
                f"Requested training config(s) are absent after filtering {episode_origin_path}: "
                f"{missing_config_filters}"
            )
    if not episode_indices:
        raise ValueError(
            "Training episode selection is empty. "
            f"source_only={source_only}, task_names={task_names}, task_configs={task_configs}"
        )
    return episode_indices


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class EpisodeSubset(Dataset[T_co]):
    """Map a compact frame index onto selected episodes in a LeRobot dataset."""

    def __init__(self, dataset: Dataset[T_co], episode_indices: Sequence[int]):
        self._dataset = dataset
        episode_data_index = getattr(dataset, "episode_data_index", None)
        if episode_data_index is None:
            raise TypeError("EpisodeSubset requires a LeRobot dataset with episode_data_index")

        episode_from = episode_data_index["from"]
        episode_to = episode_data_index["to"]
        self._base_starts: list[int] = []
        self._subset_ends: list[int] = []
        total_frames = 0
        for episode_index in episode_indices:
            start = int(episode_from[episode_index])
            end = int(episode_to[episode_index])
            self._base_starts.append(start)
            total_frames += end - start
            self._subset_ends.append(total_frames)

        self.episode_indices = tuple(episode_indices)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = int(index.__index__() if hasattr(index, "__index__") else index)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        episode_position = bisect.bisect_right(self._subset_ends, idx)
        subset_start = 0 if episode_position == 0 else self._subset_ends[episode_position - 1]
        base_index = self._base_starts[episode_position] + idx - subset_start
        return self._dataset[base_index]

    def __len__(self) -> int:
        return self._subset_ends[-1]


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    selected_episodes = None
    data_scope = os.getenv("TRAIN_DATA_SCOPE", "all").strip().lower()
    if data_scope not in {"all", "source"}:
        raise ValueError(f"Unsupported TRAIN_DATA_SCOPE={data_scope!r}; use 'all' or 'source'")

    task_names = _parse_csv_env("TRAIN_TASK_NAMES")
    task_configs = _parse_csv_env("TRAIN_TASK_CONFIGS")
    if data_scope == "source" or task_names is not None or task_configs is not None:
        selected_episodes = _selected_episode_indices(
            dataset_meta.root / "meta" / "episode_origin.jsonl",
            source_only=data_scope == "source",
            task_names=task_names,
            task_configs=task_configs,
        )

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if selected_episodes is not None:
        dataset = EpisodeSubset(dataset, selected_episodes)
        logging.info(
            "Using selected training data: scope=%s tasks=%s configs=%s episodes=%d frames=%d",
            data_scope,
            sorted(task_names) if task_names is not None else "all",
            sorted(task_configs) if task_configs is not None else "all",
            len(selected_episodes),
            len(dataset),
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    if data_config.use_caption_supervision and not data_config.use_support_context:
        raise ValueError("use_caption_supervision=True requires use_support_context=True")
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    input_transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    if data_config.use_support_context:
        if data_config.support_manifest_path is None:
            raise ValueError("use_support_context=True requires support_manifest_path")
        input_transforms.append(
            _support_context.AddSupportContext(
                data_config.support_manifest_path,
                num_support_frames=data_config.num_support_frames,
                support_cache_size=data_config.support_cache_size,
                caption_max_len=data_config.caption_max_len,
                support_chunk_size=data_config.support_chunk_size,
                support_view_override=data_config.support_view_override,
                include_caption_supervision=data_config.use_caption_supervision,
            )
        )

    return TransformedDataset(
        dataset,
        [
            *input_transforms,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    if data_config.use_support_context:
        raise NotImplementedError("Manifest-backed support context is currently only supported for LeRobot datasets.")
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    if data_config.use_support_context:
        dataset = _support_context.SupportRoundDataset(
            dataset,
            support_rounds_per_cycle=data_config.support_rounds_per_cycle,
        )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
