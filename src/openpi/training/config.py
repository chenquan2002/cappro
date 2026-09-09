"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import numpy as np
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.ur5_policy as ur5_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


_POLICY_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ROBOTWIN_ROOT = pathlib.Path(os.environ.get("ROBOTWIN_ROOT", _POLICY_ROOT.parents[1])).expanduser().resolve()
_DEFAULT_ASSETS_BASE_DIR = pathlib.Path(
    os.environ.get("OPENPI_ASSETS_BASE_DIR", _POLICY_ROOT / "assets")
).expanduser()
_DEFAULT_CHECKPOINT_BASE_DIR = pathlib.Path(
    os.environ.get("OPENPI_CHECKPOINT_BASE_DIR", _POLICY_ROOT / "checkpoints")
).expanduser()
_PI05_BASE_PARAMS = pathlib.Path(
    os.environ.get(
        "OPENPI_PI05_BASE_PARAMS",
        _ROBOTWIN_ROOT / "policy" / "pi05" / "checkpoints" / "pi05_base" / "params",
    )
).expanduser()
_LOCAL_SUPPORT_ASSETS_DIR = _POLICY_ROOT / "assets" / "pi05_aloha_robotwin_icl_random_lora"
_SIBLING_SUPPORT_ASSETS_DIR = (
    _ROBOTWIN_ROOT / "policy" / "hovapi" / "assets" / "pi05_aloha_robotwin_icl_random_lora"
)
_DEFAULT_SUPPORT_ASSETS_DIR = (
    _LOCAL_SUPPORT_ASSETS_DIR if _LOCAL_SUPPORT_ASSETS_DIR.is_dir() else _SIBLING_SUPPORT_ASSETS_DIR
)
_DEFAULT_SUPPORT_MANIFEST_PATH = (
    _ROBOTWIN_ROOT / "data" / "support_data" / "manifests" / "cappro_place_fan_phase_manifest.jsonl"
)
_SOURCE13_PHASE_EGO_MANIFEST_PATH = (
    _ROBOTWIN_ROOT / "data" / "support_data" / "manifests" / "cappro_source13_phase_ego_manifest.jsonl"
)
_SOURCE_NORM_STATS_ASSET_ID = "source_data_iclpi_repo"


def _env_bool(name: str, default: bool) -> bool:
    """Read a strict boolean environment override for experiment ablations."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    local_files_only: bool = False

    # Manifest-backed support video and train-only caption supervision.
    use_support_context: bool = False
    support_manifest_path: str | None = None
    support_rounds_per_cycle: int = 1
    support_cache_size: int = 1024
    num_support_frames: int = 8
    support_chunk_size: int = 1
    support_view_override: Literal["none", "ego", "front", "left", "right"] = "none"
    use_caption_supervision: bool = False
    caption_max_len: int = 128

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        assets_dir = epath.Path(self.assets.assets_dir or assets_dirs)
        if os.getenv("TRAIN_DATA_SCOPE", "all").strip().lower() == "source":
            assets_dir = epath.Path(os.getenv("SOURCE_NORM_STATS_DIR", str(assets_dir)))
            asset_id = os.getenv("SOURCE_NORM_STATS_ASSET_ID", _SOURCE_NORM_STATS_ASSET_ID)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(assets_dir, asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    # === 1. 参数定义 ===
    # UR5 必须开启 Delta 转换，因为 RobotWin 输出的是绝对位置
    use_delta_joint_actions: bool = True
    
    # 默认提示词，以防数据集缺失
    default_prompt: str | None = None
    adapt_to_pi: bool = False
    # === 2. Repack Transforms (默认值) ===
    # 这里定义了如何从 LeRobot 数据集映射到 Policy 需要的名称
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            # 左边: Policy 需要的名字 | 右边: RobotWin HDF5 里的名字
                            "base_rgb": "observation.images.cam_high",
                            "wrist_rgb": "observation.images.cam_left_wrist", # 请确认 HDF5 是否叫 cam_wrist
                        },
                        # 状态: RobotWin 已合并 state (7维)
                        "state": "observation.state",
                        # 动作
                        "actions": "action",
                        # 指令
                        "prompt": "task",
                    }
                )
            ]
        )
    )

    # 指定 Data Loader 从哪个 key 读取动作序列，这里对应上面 Repack 里的 "actions"
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # === 3. Data Transforms ===
        # 使用我们之前写好的 UR5Inputs / UR5Outputs
        # 注意：这里我们移除了 adapt_to_pi 参数，因为 UR5Inputs 实现里没用到
        data_transforms = _transforms.Group(
            inputs=[ur5_policy.UR5Inputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.UR5Outputs()],
        )

        # === 4. Delta 转换逻辑 ===
        if self.use_delta_joint_actions:
            # 关键修改：UR5e 是 6 个关节 + 1 个夹爪
            # (6, -1) 表示前 6 个维度为 True (做 Delta)，最后 1 个维度为 False (保持绝对值)
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # === 5. Model Transforms ===
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # 返回最终配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms, # 使用上面定义的类属性
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
##################################################acp##################################
class InjectACPTagFromIndicator:
    """Turn (prompt/task, acp_indicator) into the final prompt string.

    This transform runs in `data_transforms.inputs` BEFORE `AlohaInputs`.
    It only rewrites `prompt`; policy/model code stays unchanged.
    """

    def __init__(
        self,
        positive_tag: str = "Advantage: positive",
        negative_tag: str = "Advantage: negative",
        separator: str = "\n",
        drop_indicator_key: bool = True,
    ):
        self.positive_tag = positive_tag
        self.negative_tag = negative_tag
        self.separator = separator
        self.drop_indicator_key = drop_indicator_key

    def _scalar_to_int(self, x: Any) -> int:
        if x is None:
            return 0
        if isinstance(x, (int, np.integer, bool, np.bool_)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return 0
            return int(np.asarray(x).reshape(-1)[0])
        if isinstance(x, (list, tuple)):
            if not x:
                return 0
            return self._scalar_to_int(x[0])
        # Handle torch/jax arrays conservatively without importing them.
        if hasattr(x, "shape") and hasattr(x, "reshape"):
            try:
                return int(x.reshape(-1)[0])
            except Exception:
                pass
        return int(x)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        prompt = data.get("prompt")
        if prompt is None:
            prompt = data.get("task", "")
        indicator = self._scalar_to_int(data.get("acp_indicator", 0))
        tag = self.positive_tag if indicator == 1 else self.negative_tag
        data["prompt"] = f"{prompt}{self.separator}{tag}" if prompt else tag
        if self.drop_indicator_key:
            data.pop("acp_indicator", None)
        return data


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaACPDataConfig(DataConfigFactory):
    """Aloha rollout/ACP data config.

    Differences vs. `LeRobotAlohaDataConfig`:
    - repack keeps `acp_indicator`
    - data transforms inject ACP tag into `prompt` before `AlohaInputs`
    - policy/model code remains unchanged
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        # prompt is expected to be created from task by prompt_from_task=True
                        "prompt": "prompt",
                        # keep acp_indicator for the ACP prompt transform
                        "acp_indicator": "acp_indicator",
                    }
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs, model_config: _model.BaseModelConfig):
        data_transforms = _transforms.Group(
            inputs=[
                InjectACPTagFromIndicator(),
                aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi),
            ],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


##################################################acp##################################
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
# 引入刚才定义的 policy 文件
import openpi.policies.aloha_policy_ee as aloha_policy_ee
############### add +++++++++++++
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig_ee(DataConfigFactory):
    # 如果为 True，将应用 EE Delta 转换 (16 dim -> 14 dim)
    use_delta_ee_actions: bool = True
    default_prompt: str | None = None
    
    # Repack: 映射 LeRobot Dataset 的 Key 到 Policy 的 Key
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state", # 16 dim
                        "actions": "action",          # 16 dim
                        "prompt": "task",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. 基本 Input Transform
        inputs_list = [aloha_policy_ee.AlohaInputs_EE()]
        outputs_list = [aloha_policy_ee.AlohaOutputs_EE()]
        # 2. 注入 Delta 转换逻辑
        # 注意：这里不用 transforms.DeltaActions，而是用我们在 policy_aloha_ee 里写的专用转换
        if self.use_delta_ee_actions:
            # Input pipeline: Absolute(16) -> Delta(14) -> Model
            inputs_list.append(aloha_policy_ee.EEDeltaActions())
            outputs_list.insert(0, aloha_policy_ee.EEAbsoluteActions())
        data_transforms = _transforms.Group(
            inputs=inputs_list,
            outputs=outputs_list,
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = str(_DEFAULT_ASSETS_BASE_DIR)
    # Base directory for checkpoints.
    checkpoint_base_dir: str = str(_DEFAULT_CHECKPOINT_BASE_DIR)

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ##add+++++++++++++++++++++aloha-ee
    TrainConfig(
        name="pi05_aloha_robotwin_ee_lora",
        # 核心：定义模型
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 【极重要】修改 Action Dimension
            # 原始 State/Action 是 16维，但经过 Delta 转换后变成了 14维 (XYZ+Euler+Grip)*2
            # 所以模型需要学习的是 14维 的分布
            # action_dim=14, 
        ),
        
        # 数据配置
        data=LeRobotAlohaDataConfig_ee(
            # 指向你刚刚转换好的 LeRobot 数据集
            repo_id=os.getenv("REPO_ID", "chenquan/beat_block_ee_test"), 
            
            # 使用 Delta 转换
            use_delta_ee_actions=True, 
            
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, 
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),
        
        batch_size=32,
        num_train_steps=30000,
        ema_decay=None, 
        fsdp_devices=2,
        save_interval=10000,
    ),
    ###
    ### finetune config for robotwin
    ###
    ##ur5e
    # 你的 RobotWin UR5e 专用配置
    TrainConfig(
        # 1. 修改名称：这个名字将决定 assets 文件夹生成在哪里
        # 运行 compute_norm_stats 后，统计数据会自动生成在 ./assets/pi05_ur5_robotwin_lora/ 下
        name="pi05_ur5_robotwin_lora",
        
        # 2. 模型定义：保持原样 (Pi0.5 + LoRA)
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ),
        
        # 3. 数据配置：切换为我们新写的 UR5 配置类
        data=LeRobotUR5DataConfig(
            # 替换为你实际转换出来的 LeRobot 数据集路径
            repo_id=os.getenv("REPO_ID", "pick_blue_cube_test_3_repo"),
            # repo_id="pick_place_demo1_repo", 
            adapt_to_pi=False,
            # 【关键】：我删除了 assets=AssetsConfig(...) 代码块
            # 这样代码就会使用默认逻辑：去 ./assets/pi05_ur5_robotwin_lora/repo_id/ 下面找数据
            
            # 4. 字段映射：把 RobotWin 的数据映射给 UR5Inputs
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    # 图像：右边是 RobotWin HDF5 里的名字，左边是 UR5Inputs 需要的名字
                    "base_rgb": "observation.images.cam_high",
                    "wrist_rgb": "observation.images.cam_left_wrist", # 请确认 RobotWin 里是不是叫 cam_wrist
                    
                    # 状态：直接映射
                    "state": "observation.state",
                    
                    # 动作
                    "actions": "action",
                    
                    # 文本指令：RobotWin 转换时存在了 task 字段里
                    "prompt": "task",
                })
            ]),
            
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        
        # 5. 冻结参数：保持原样
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, 
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        
        # 6. 权重加载：保持原样 (指向 Pi0.5 Base)
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),
        
        # 7. 训练超参：保持原样
        batch_size=32,
        num_train_steps=20000,  
        ema_decay=None,
        fsdp_devices=2,
        save_interval=5000,
    ),
    #########acp train config for robotwin##############
    TrainConfig(
        name="pi05_aloha_robotwin_acp_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaACPDataConfig(
            repo_id=os.getenv("REPO_ID", "place_mouse_pad_test_1_repo"),
            adapt_to_pi=False,
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),
        batch_size=32,
        num_train_steps=30000,
        ema_decay=None,
        fsdp_devices=2,
        save_interval=10000,
    ),
    #######################################################
    TrainConfig(
        name="pi05_aloha_robotwin_lora",  # 给个新名字
        
        # === 核心模型定义 ===
        model=pi0_config.Pi0Config(
            pi05=True,  # 关键：启用 Pi0.5 架构
            paligemma_variant="gemma_2b_lora",      # 关键：加载带 LoRA 的 Paligemma
            action_expert_variant="gemma_300m_lora" # 关键：加载带 LoRA 的 Action Expert
        ),
        
        # === 数据配置 (根据你的实际情况) ===
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "place_mouse_pad_test_1_repo"),  # 替换为你实际转换出来的 LeRobot 数据集路径
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        
        # === 冻结参数设置 ===
        # 注意：这里必须和上面 model 的定义保持一致，这样才能正确生成“冻结非LoRA参数”的过滤器
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, 
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        
        # === 权重加载 ===
        # 注意：这里必须加载 pi05_base 的权重，而不是 pi0_base
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_PI05_BASE_PARAMS)),
        
        # === 其他 LoRA 训练惯例 ===
        batch_size=32,
        num_train_steps=200000,  
        ema_decay=None, # LoRA 训练通常关闭 EMA 以节省显存
        fsdp_devices=2,
        save_interval=10000,
    ),
    # CapPro: support-video caption supervision with semantic query tokens
    # passed directly to the action prefix.
    TrainConfig(
        name="pi05_aloha_robotwin_cappro_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            num_support_frames=8,
            use_support_progress=True,
            use_support_role_embedding=True,
            support_progress_embed_dim=128,
            use_support_token_compression=_env_bool("USE_SUPPORT_TOKEN_COMPRESSION", True),
            support_static_tokens=64,
            support_motion_tokens_per_frame=24,
            support_compression_temperature=1.0,
            use_caption_supervision=True,
            caption_max_len=int(os.getenv("CAPTION_MAX_LEN", "160")),
            caption_decode_chunk_size=4,
            caption_loss_weight=float(os.getenv("CAPTION_LOSS_WEIGHT", "0.1")),
            num_caption_queries=int(os.getenv("NUM_CAPTION_QUERIES", "4")),
            caption_action_gate_init=float(os.getenv("CAPTION_ACTION_GATE_INIT", "0.1")),
            caption_robot_tokens_per_image=int(os.getenv("CAPTION_ROBOT_TOKENS_PER_IMAGE", "64")),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "source_data_hovapi_repo"),
            assets=AssetsConfig(
                assets_dir=os.getenv(
                    "SUPPORT_ASSETS_DIR",
                    str(_DEFAULT_SUPPORT_ASSETS_DIR),
                ),
                asset_id=os.getenv("ASSET_ID", "source_data_hovapi_repo"),
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                use_support_context=True,
                support_manifest_path=os.getenv(
                    "SUPPORT_MANIFEST_PATH",
                    str(_DEFAULT_SUPPORT_MANIFEST_PATH),
                ),
                support_rounds_per_cycle=int(os.getenv("SUPPORT_ROUNDS_PER_CYCLE", "20")),
                support_cache_size=1024,
                num_support_frames=8,
                support_chunk_size=int(os.getenv("SUPPORT_CHUNK_SIZE", "1")),
                support_view_override=os.getenv("SUPPORT_VIEW_OVERRIDE", "none").strip().lower(),
                use_caption_supervision=True,
                caption_max_len=int(os.getenv("CAPTION_MAX_LEN", "160")),
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            use_caption_supervision=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),
        # Global batch size. With two visible GPUs and fsdp_devices=2, this is two samples per GPU.
        batch_size=96,
        num_train_steps=200_000,
        ema_decay=None,
        fsdp_devices=1,
        save_interval=10_000
    ),
    # CapPro full source13 phase-caption training with ego support videos.
    TrainConfig(
        name="pi05_aloha_robotwin_cappro_source13_full_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            num_support_frames=8,
            use_support_progress=True,
            use_support_role_embedding=True,
            support_progress_embed_dim=128,
            use_support_token_compression=True,
            support_static_tokens=64,
            support_motion_tokens_per_frame=24,
            support_compression_temperature=1.0,
            use_caption_supervision=True,
            caption_max_len=96,
            caption_decode_chunk_size=4,
            caption_loss_weight=0.1,
            num_caption_queries=4,
            caption_action_gate_init=0.1,
            caption_robot_tokens_per_image=64,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "source_data_hovapi_repo"),
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_SUPPORT_ASSETS_DIR),
                asset_id="source_data_iclpi_repo",
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                use_support_context=True,
                support_manifest_path=os.getenv(
                    "SUPPORT_MANIFEST_PATH",
                    str(_SOURCE13_PHASE_EGO_MANIFEST_PATH),
                ),
                support_rounds_per_cycle=5,
                support_cache_size=1024,
                num_support_frames=8,
                support_chunk_size=1,
                support_view_override="ego",
                use_caption_supervision=True,
                caption_max_len=96,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            use_caption_supervision=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_PI05_BASE_PARAMS)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        seed=42,
        batch_size=128,
        num_workers=2,
        num_train_steps=200_000,
        log_interval=100,
        save_interval=5_000,
        keep_period=50_000,
        ema_decay=None,
        wandb_enabled=True,
        fsdp_devices=1,
    ),
    # CapPro source13 GridS training without robot chunk-progress input in the
    # action prefix. Chunk progress remains in the observation/data bookkeeping
    # fields, but is not passed to action inference.
    TrainConfig(
        name="pi05_aloha_robotwin_cappro_source13_grids_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            num_support_frames=8,
            use_support_progress=True,
            use_support_role_embedding=True,
            support_progress_embed_dim=128,
            use_support_grid_sampling=True,
            support_grid_hidden_dim=int(os.getenv("SUPPORT_GRID_HIDDEN_DIM", "256")),
            support_grid_tokens_per_frame=int(os.getenv("SUPPORT_GRID_TOKENS_PER_FRAME", "32")),
            use_support_token_compression=False,
            use_caption_supervision=True,
            caption_max_len=96,
            caption_decode_chunk_size=4,
            caption_loss_weight=0.1,
            num_caption_queries=4,
            caption_action_gate_init=0.1,
            caption_robot_tokens_per_image=64,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "source_data_hovapi_repo"),
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_SUPPORT_ASSETS_DIR),
                asset_id="source_data_iclpi_repo",
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                use_support_context=True,
                support_manifest_path=os.getenv(
                    "SUPPORT_MANIFEST_PATH",
                    str(_SOURCE13_PHASE_EGO_MANIFEST_PATH),
                ),
                support_rounds_per_cycle=5,
                support_cache_size=1024,
                num_support_frames=8,
                support_chunk_size=1,
                support_view_override="ego",
                use_caption_supervision=True,
                caption_max_len=96,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            use_caption_supervision=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_PI05_BASE_PARAMS)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=200_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        seed=42,
        batch_size=128,
        num_workers=2,
        num_train_steps=200_000,
        log_interval=100,
        save_interval=5_000,
        keep_period=50_000,
        ema_decay=None,
        wandb_enabled=True,
        fsdp_devices=1,
    ),
    # CapPro single-task phase-caption training for place_fan/demo_clean.
    TrainConfig(
        name="pi05_aloha_robotwin_cappro_place_fan_clean_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            num_support_frames=8,
            use_support_progress=True,
            use_support_role_embedding=True,
            support_progress_embed_dim=128,
            use_support_token_compression=True,
            support_static_tokens=64,
            support_motion_tokens_per_frame=24,
            support_compression_temperature=1.0,
            use_caption_supervision=True,
            caption_max_len=96,
            caption_decode_chunk_size=4,
            caption_loss_weight=0.1,
            num_caption_queries=4,
            caption_action_gate_init=0.1,
            caption_robot_tokens_per_image=64,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "source_data_hovapi_repo"),
            # Reuse the source-task stats already computed for this robot/data family.
            # The config name controls compute_norm_stats output, but this explicit
            # AssetsConfig controls which existing stats training reads.
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_SUPPORT_ASSETS_DIR),
                asset_id="source_data_iclpi_repo",
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                use_support_context=True,
                support_manifest_path=os.getenv(
                    "SUPPORT_MANIFEST_PATH",
                    str(_DEFAULT_SUPPORT_MANIFEST_PATH),
                ),
                support_rounds_per_cycle=20,
                support_cache_size=1024,
                num_support_frames=8,
                support_chunk_size=1,
                # This experiment always trains from the ego human-video view.
                # Non-ego manifest records are remapped to the matching ego path.
                support_view_override="ego",
                use_caption_supervision=True,
                caption_max_len=96,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            use_caption_supervision=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_PI05_BASE_PARAMS)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        seed=42,
        batch_size=64,
        num_workers=2,
        num_train_steps=7_500,
        log_interval=100,
        save_interval=5_000,
        keep_period=5_000,
        ema_decay=None,
        wandb_enabled=True,
        # Two visible H100s form two data-parallel replicas; the model is not sharded.
        fsdp_devices=1,
    ),
    # GridS version of the place_fan/demo_clean CapPro experiment above. It
    # starts from the same pi0.5 base parameters as the heuristic baseline so
    # the compression method is the controlled experimental variable.
    TrainConfig(
        name="pi05_aloha_robotwin_cappro_place_fan_clean_grids_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            num_support_frames=8,
            use_support_progress=True,
            use_support_role_embedding=True,
            support_progress_embed_dim=128,
            use_support_grid_sampling=True,
            support_grid_hidden_dim=int(os.getenv("SUPPORT_GRID_HIDDEN_DIM", "256")),
            support_grid_tokens_per_frame=int(os.getenv("SUPPORT_GRID_TOKENS_PER_FRAME", "32")),
            use_support_token_compression=False,
            use_caption_supervision=True,
            caption_max_len=96,
            caption_decode_chunk_size=4,
            caption_loss_weight=0.1,
            num_caption_queries=4,
            caption_action_gate_init=0.1,
            caption_robot_tokens_per_image=64,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "source_data_hovapi_repo"),
            assets=AssetsConfig(
                assets_dir=str(_DEFAULT_SUPPORT_ASSETS_DIR),
                asset_id="source_data_iclpi_repo",
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                use_support_context=True,
                support_manifest_path=os.getenv(
                    "SUPPORT_MANIFEST_PATH",
                    str(_DEFAULT_SUPPORT_MANIFEST_PATH),
                ),
                support_rounds_per_cycle=20,
                support_cache_size=1024,
                num_support_frames=8,
                support_chunk_size=1,
                support_view_override="ego",
                use_caption_supervision=True,
                caption_max_len=96,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_support_context=True,
            use_caption_supervision=True,
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_PI05_BASE_PARAMS)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        seed=42,
        batch_size=64,
        num_workers=2,
        num_train_steps=7_500,
        log_interval=100,
        save_interval=5_000,
        keep_period=5_000,
        ema_decay=None,
        wandb_enabled=True,
        # Two visible H100s form two data-parallel replicas; the model is not sharded.
        fsdp_devices=1,
    ),
    ##### vlm freezed + action expert full finetune
    TrainConfig(
        name="pi05_aloha_robotwin_head_only",

        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b",       # VLM 不用 LoRA
            action_expert_variant="gemma_300m" # 动作头全量训练
        ),

        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "place_mouse_pad_test_1_repo"),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),

        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m"
        ).get_vlm_freeze_filter(),

        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),

        batch_size=32,
        num_train_steps=30000,
        ema_decay=0.99,
        fsdp_devices=2,
        save_interval=10000,
    ),
    TrainConfig(
        name="pi05_aloha_robotwin_base_eval",

        # === 核心模型定义 ===
        model=pi0_config.Pi0Config(
            pi05=True,                 # 仍然是 Pi0.5
            paligemma_variant="gemma_2b",      # 非 LoRA
            action_expert_variant="gemma_300m" # 非 LoRA
        ),

        # === 数据配置：完全复用你现在的 ALOHA / Robotwin 数据设置 ===
        data=LeRobotAlohaDataConfig(
            repo_id=os.getenv("REPO_ID", "place_mouse_pad_test_1_repo"),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),

        # === 纯 base 测评不需要冻结器 ===
        # 因为你现在是拿现成 checkpoint 做 eval，不是继续训练
        freeze_filter=nnx.Nothing,

        # === 这里写 pi05_base 的路径，方便和你现有风格保持一致 ===
        # 但如果你走的是 serve / eval 从 checkpoint_dir 直接加载，
        # 这项实际上不是关键
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(_PI05_BASE_PARAMS)
        ),

        # === 下面这些对 eval 基本无所谓，给占位即可 ===
        batch_size=32,
        num_train_steps=30000,
        ema_decay=None,
        fsdp_devices=2,
        save_interval=10000,
    ),
    TrainConfig(
        name="pi05_aloha_full_base",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_base by lora
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="test",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_fast_base by lora
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=2,  # refer line 359
    ),
    # pi0_base by full
    TrainConfig(
        name="pi0_base_aloha_robotwin_full",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=4,  # refer line 359
    ),
    # pi0_fast_base by full
    TrainConfig(
        name="pi0_fast_aloha_robotwin_full",
        model=pi0_fast.Pi0FASTConfig(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
