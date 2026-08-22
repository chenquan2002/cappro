import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Manifest-backed support video configuration.
    use_support_context: bool = False
    num_support_frames: int = 8
    use_support_progress: bool = True
    use_support_role_embedding: bool = True
    support_progress_embed_dim: int = 128

    # 8 frames x 256 SigLIP tokens are compressed to 64 + 8 x 24 tokens.
    use_support_token_compression: bool = True
    support_static_tokens: int = 64
    support_motion_tokens_per_frame: int = 24
    support_compression_temperature: float = 1.0

    # Caption supervision and semantic query tokens. Caption input tokens remain
    # train-only; the query outputs can be passed to the action prefix.
    use_caption_supervision: bool = False
    caption_max_len: int = 128
    caption_decode_chunk_size: int = 4
    caption_loss_weight: float = 1.0
    num_caption_queries: int = 4
    caption_action_gate_init: float = 0.1
    # Pool each current robot view from 16x16 SigLIP tokens to 8x8 for
    # caption/query inference. The action prefix keeps all 256 tokens.
    caption_robot_tokens_per_image: int = 64

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.num_support_frames < 1:
            raise ValueError("num_support_frames must be >= 1")
        if self.support_progress_embed_dim < 2 or self.support_progress_embed_dim % 2 != 0:
            raise ValueError("support_progress_embed_dim must be a positive even number")
        static_grid_size = math.isqrt(self.support_static_tokens)
        if static_grid_size * static_grid_size != self.support_static_tokens:
            raise ValueError("support_static_tokens must be a square number")
        if not 1 <= self.support_motion_tokens_per_frame <= 256:
            raise ValueError("support_motion_tokens_per_frame must be in [1, 256]")
        if self.support_compression_temperature <= 0:
            raise ValueError("support_compression_temperature must be > 0")
        if self.use_caption_supervision and not self.use_support_context:
            raise ValueError("use_caption_supervision=True requires use_support_context=True")
        if self.caption_max_len < 1:
            raise ValueError("caption_max_len must be >= 1")
        if self.caption_decode_chunk_size < 1:
            raise ValueError("caption_decode_chunk_size must be >= 1")
        if self.caption_loss_weight < 0:
            raise ValueError("caption_loss_weight must be >= 0")
        if self.num_caption_queries < 1:
            raise ValueError("num_caption_queries must be >= 1")
        if not 0.0 <= self.caption_action_gate_init <= 1.0:
            raise ValueError("caption_action_gate_init must be in [0, 1]")
        caption_robot_grid_size = math.isqrt(self.caption_robot_tokens_per_image)
        if (
            caption_robot_grid_size * caption_robot_grid_size != self.caption_robot_tokens_per_image
            or caption_robot_grid_size < 1
            or 16 % caption_robot_grid_size != 0
        ):
            raise ValueError("caption_robot_tokens_per_image must be a square grid that evenly divides 16x16")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                support_images=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.num_support_frames, *_model.IMAGE_RESOLUTION, 3],
                        jnp.float32,
                    )
                    if self.use_support_context
                    else None
                ),
                support_image_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.num_support_frames], jnp.bool_)
                    if self.use_support_context
                    else None
                ),
                support_frame_progress=(
                    jax.ShapeDtypeStruct([batch_size, self.num_support_frames], jnp.float32)
                    if self.use_support_context
                    else None
                ),
                chunk_progress=(
                    jax.ShapeDtypeStruct([batch_size, 1], jnp.float32) if self.use_support_context else None
                ),
                caption_input_tokens=(
                    jax.ShapeDtypeStruct([batch_size, self.caption_max_len], jnp.int32)
                    if self.use_caption_supervision
                    else None
                ),
                caption_input_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.caption_max_len], jnp.bool_)
                    if self.use_caption_supervision
                    else None
                ),
                caption_target_tokens=(
                    jax.ShapeDtypeStruct([batch_size, self.caption_max_len], jnp.int32)
                    if self.use_caption_supervision
                    else None
                ),
                caption_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.caption_max_len], jnp.bool_)
                    if self.use_caption_supervision
                    else None
                ),
                caption_hand_side_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.caption_max_len], jnp.bool_)
                    if self.use_caption_supervision
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    # modify ==> freeze the vlm except expert action head.
    def get_vlm_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze VLM / PaliGemma params, keep action expert trainable."""
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        # Freeze all LLM parameters except the action expert branch.
        return nnx.All(
            gemma_params_filter,
            nnx.Not(action_expert_params_filter),
        )
