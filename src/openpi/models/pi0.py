from collections.abc import Callable
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_caption_bottleneck_attn_mask(input_mask, *, context_len: int, query_len: int):
    """Build attention for visual context, semantic queries, and causal captions.

    Context and query tokens form one bidirectional prefix and cannot read the
    later teacher-forcing caption. Caption tokens can read the semantic queries
    and their causal caption history, but not the raw visual context. This makes
    the queries the only visual path into caption prediction.
    """
    sequence_len = input_mask.shape[1]
    query_end = context_len + query_len
    if context_len < 1 or query_len < 1 or query_end > sequence_len:
        raise ValueError(
            f"Invalid caption attention segments: context_len={context_len}, "
            f"query_len={query_len}, sequence_len={sequence_len}"
        )

    row = jnp.arange(sequence_len)[:, None]
    column = jnp.arange(sequence_len)[None, :]
    prefix_attention = jnp.logical_and(row < query_end, column < query_end)
    caption_to_query = jnp.logical_and(
        row >= query_end,
        jnp.logical_and(column >= context_len, column < query_end),
    )
    caption_causal = jnp.logical_and(
        row >= query_end,
        jnp.logical_and(column >= query_end, column <= row),
    )
    attention = jnp.logical_or(prefix_attention, jnp.logical_or(caption_to_query, caption_causal))
    valid_mask = jnp.logical_and(input_mask[:, None, :], input_mask[:, :, None])
    return jnp.logical_and(attention[None, :, :], valid_mask)


def ensure_nonempty_caption_input_mask(prefix_token_mask, caption_input_mask):
    """Make fully masked caption-prefix rows safe without adding supervision."""
    if caption_input_mask.shape[1] < 1:
        raise ValueError("caption_input_mask must contain at least one position")
    caption_input_mask = caption_input_mask.astype(jnp.bool_)
    has_any_input = jnp.logical_or(
        jnp.any(prefix_token_mask, axis=1),
        jnp.any(caption_input_mask, axis=1),
    )
    return caption_input_mask.at[:, 0].set(jnp.logical_or(caption_input_mask[:, 0], jnp.logical_not(has_any_input)))


def compute_caption_token_metrics(
    caption_hidden,
    target_tokens,
    loss_mask,
    hand_side_mask,
    *,
    decode_fn: Callable,
    chunk_size: int,
):
    """Decode caption states in time chunks and aggregate per-sample metrics."""
    if caption_hidden.shape[:2] != target_tokens.shape:
        raise ValueError(
            f"Caption hidden states and targets must share [B, T], got {caption_hidden.shape} and {target_tokens.shape}"
        )
    if loss_mask.shape != target_tokens.shape or hand_side_mask.shape != target_tokens.shape:
        raise ValueError("Caption targets, loss mask, and hand-side mask must have the same shape")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    batch_size, caption_len = target_tokens.shape
    loss_sum = jnp.zeros((batch_size,), dtype=jnp.float32)
    correct_sum = jnp.zeros((batch_size,), dtype=jnp.float32)
    token_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    hand_loss_sum = jnp.zeros((batch_size,), dtype=jnp.float32)
    hand_correct_sum = jnp.zeros((batch_size,), dtype=jnp.float32)
    hand_token_count = jnp.zeros((batch_size,), dtype=jnp.int32)

    for start in range(0, caption_len, chunk_size):
        end = min(start + chunk_size, caption_len)
        logits = decode_fn(caption_hidden[:, start:end]).astype(jnp.float32)
        chunk_targets = target_tokens[:, start:end]
        chunk_loss_mask = loss_mask[:, start:end].astype(jnp.bool_)
        chunk_hand_mask = jnp.logical_and(hand_side_mask[:, start:end], chunk_loss_mask)

        target_logits = jnp.take_along_axis(logits, chunk_targets[..., None], axis=-1)[..., 0]
        token_nll = jax.nn.logsumexp(logits, axis=-1) - target_logits
        token_correct = jnp.argmax(logits, axis=-1) == chunk_targets

        loss_sum += jnp.sum(token_nll * chunk_loss_mask, axis=1)
        correct_sum += jnp.sum(token_correct * chunk_loss_mask, axis=1)
        token_count += jnp.sum(chunk_loss_mask, axis=1, dtype=jnp.int32)
        hand_loss_sum += jnp.sum(token_nll * chunk_hand_mask, axis=1)
        hand_correct_sum += jnp.sum(token_correct * chunk_hand_mask, axis=1)
        hand_token_count += jnp.sum(chunk_hand_mask, axis=1, dtype=jnp.int32)

    token_denominator = jnp.maximum(token_count, 1).astype(jnp.float32)
    hand_denominator = jnp.maximum(hand_token_count, 1).astype(jnp.float32)
    return {
        "caption_loss": loss_sum / token_denominator,
        "caption_token_accuracy": correct_sum / token_denominator,
        "caption_hand_side_loss": hand_loss_sum / hand_denominator,
        "caption_hand_side_accuracy": hand_correct_sum / hand_denominator,
        "caption_token_count": token_count,
        "caption_hand_side_token_count": hand_token_count,
    }


def aggregate_caption_metrics(per_sample_metrics):
    """Aggregate per-sample caption means without diluting padded rows."""
    token_count = per_sample_metrics["caption_token_count"]
    hand_token_count = per_sample_metrics["caption_hand_side_token_count"]
    total_token_count = jnp.sum(token_count)
    total_hand_token_count = jnp.sum(hand_token_count)
    token_denominator = jnp.maximum(total_token_count, 1).astype(jnp.float32)
    hand_denominator = jnp.maximum(total_hand_token_count, 1).astype(jnp.float32)

    return {
        "caption_loss": jnp.sum(per_sample_metrics["caption_loss"] * token_count) / token_denominator,
        "caption_token_accuracy": (
            jnp.sum(per_sample_metrics["caption_token_accuracy"] * token_count) / token_denominator
        ),
        "caption_hand_side_loss": (
            jnp.sum(per_sample_metrics["caption_hand_side_loss"] * hand_token_count) / hand_denominator
        ),
        "caption_hand_side_accuracy": (
            jnp.sum(per_sample_metrics["caption_hand_side_accuracy"] * hand_token_count) / hand_denominator
        ),
        "caption_token_count": total_token_count,
        "caption_hand_side_token_count": total_hand_token_count,
        "caption_valid_samples": jnp.sum(token_count > 0),
    }


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def compress_support_image_tokens(
    value_tokens,
    score_tokens,
    support_image_mask,
    *,
    num_static_tokens: int,
    motion_tokens_per_frame: int,
    temperature: float,
):
    """Keep low-motion scene context and high-motion per-frame details.

    ``score_tokens`` must contain raw visual features. ``value_tokens`` may
    additionally contain progress embeddings; separating them prevents temporal
    metadata from changing which spatial patches are considered motion.
    """
    if value_tokens.shape != score_tokens.shape:
        raise ValueError(f"value_tokens and score_tokens must match, got {value_tokens.shape} and {score_tokens.shape}")

    batch_size, num_frames, tokens_per_frame, _ = value_tokens.shape
    grid_size = int(tokens_per_frame**0.5)
    if grid_size * grid_size != tokens_per_frame:
        raise ValueError(f"Support image tokens must form a square grid, got {tokens_per_frame}")
    static_grid_size = int(num_static_tokens**0.5)
    if static_grid_size * static_grid_size != num_static_tokens:
        raise ValueError(f"num_static_tokens must be a square number, got {num_static_tokens}")
    if grid_size % static_grid_size != 0:
        raise ValueError(f"Cannot pool {grid_size}x{grid_size} tokens to {static_grid_size}x{static_grid_size}")
    if not 1 <= motion_tokens_per_frame <= tokens_per_frame:
        raise ValueError(f"motion_tokens_per_frame must be in [1, {tokens_per_frame}]")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    support_image_mask = support_image_mask.astype(jnp.bool_)
    values = value_tokens.astype(jnp.float32)
    scores = jax.lax.stop_gradient(score_tokens.astype(jnp.float32))

    if num_frames == 1:
        motion = jnp.zeros((batch_size, 1, tokens_per_frame), dtype=jnp.float32)
    else:
        frame_diff = jnp.mean(jnp.square(scores[:, 1:] - scores[:, :-1]), axis=-1)
        # The first frame keeps the same moving regions as the first transition,
        # preserving before/after features for those regions.
        motion = jnp.concatenate([frame_diff[:, :1], frame_diff], axis=1)
    motion = jax.lax.stop_gradient(motion)

    value_grid = einops.rearrange(values, "b k (h w) d -> b k h w d", h=grid_size, w=grid_size)
    motion_grid = einops.rearrange(motion, "b k (h w) -> b k h w", h=grid_size, w=grid_size)
    valid_grid = support_image_mask.astype(jnp.float32)[:, :, None, None]

    motion_sum = jnp.sum(motion_grid * valid_grid, axis=(1, 2, 3))
    motion_count = jnp.sum(valid_grid, axis=(1, 2, 3)) * tokens_per_frame
    mean_motion = (motion_sum / (motion_count + 1e-6))[:, None, None, None, None]
    static_weight = jnp.exp(-motion_grid[:, :, :, :, None] / (mean_motion + 1e-6) / temperature)
    static_weight = jax.lax.stop_gradient(static_weight * valid_grid[:, :, :, :, None])

    static_denom = jnp.sum(static_weight, axis=1) + 1e-6
    static_context = jnp.sum(value_grid * static_weight, axis=1) / static_denom
    pool = grid_size // static_grid_size
    static_tokens = einops.rearrange(
        static_context,
        "b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d",
        gh=static_grid_size,
        gw=static_grid_size,
        ph=pool,
        pw=pool,
    )
    static_tokens = jnp.mean(static_tokens, axis=2)
    static_mask = jnp.broadcast_to(
        jnp.any(support_image_mask, axis=1, keepdims=True),
        (batch_size, num_static_tokens),
    )

    motion_for_topk = jnp.where(
        support_image_mask[:, :, None],
        motion,
        jnp.full_like(motion, -1.0e9),
    )
    _, top_indices = jax.lax.top_k(motion_for_topk, motion_tokens_per_frame)
    motion_tokens = jnp.take_along_axis(value_tokens, top_indices[..., None], axis=2)
    motion_tokens = einops.rearrange(motion_tokens, "b k m d -> b (k m) d")
    motion_mask = einops.repeat(
        support_image_mask,
        "b k -> b (k m)",
        m=motion_tokens_per_frame,
    )

    compressed_tokens = jnp.concatenate(
        [static_tokens.astype(value_tokens.dtype), motion_tokens],
        axis=1,
    )
    compressed_mask = jnp.concatenate([static_mask, motion_mask], axis=1)
    return compressed_tokens, compressed_mask


def spatial_pool_image_tokens(tokens, *, num_tokens: int):
    """Average-pool one square image-token grid to a smaller square grid."""
    tokens_per_image = tokens.shape[1]
    source_grid_size = int(tokens_per_image**0.5)
    target_grid_size = int(num_tokens**0.5)
    if source_grid_size * source_grid_size != tokens_per_image:
        raise ValueError(f"Image tokens must form a square grid, got {tokens_per_image}")
    if target_grid_size * target_grid_size != num_tokens or target_grid_size < 1:
        raise ValueError(f"num_tokens must be a positive square number, got {num_tokens}")
    if source_grid_size % target_grid_size != 0:
        raise ValueError(
            f"Cannot pool {source_grid_size}x{source_grid_size} tokens to "
            f"{target_grid_size}x{target_grid_size}"
        )
    if tokens_per_image == num_tokens:
        return tokens

    dtype = tokens.dtype
    pool_size = source_grid_size // target_grid_size
    grid = einops.rearrange(
        tokens.astype(jnp.float32),
        "b (h w) d -> b h w d",
        h=source_grid_size,
        w=source_grid_size,
    )
    pooled = einops.rearrange(
        grid,
        "b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d",
        gh=target_grid_size,
        gw=target_grid_size,
        ph=pool_size,
        pw=pool_size,
    )
    return jnp.mean(pooled, axis=2).astype(dtype)


def bilinear_sample_image_tokens(tokens, coordinates):
    """Differentiably sample a square visual-token grid at continuous points.

    Args:
        tokens: Float array with shape ``[B, K, H, W, D]``.
        coordinates: Float array with shape ``[B, K, S, 2]``. The final
            dimension is ``(x, y)`` in normalized ``[0, 1]`` coordinates.

    Returns:
        Sampled tokens with shape ``[B, K, S, D]``. The interpolation weights
        are differentiable with respect to ``coordinates``; this is the path
        through which the action loss trains a GridS coordinate predictor.
    """
    if tokens.ndim != 5 or coordinates.ndim != 4:
        raise ValueError(
            f"Expected tokens [B,K,H,W,D] and coordinates [B,K,S,2], "
            f"got {tokens.shape} and {coordinates.shape}"
        )
    if tokens.shape[:2] != coordinates.shape[:2] or coordinates.shape[-1] != 2:
        raise ValueError(
            f"Token/frame dimensions and coordinate dimensions must match, "
            f"got {tokens.shape} and {coordinates.shape}"
        )

    batch_size, num_frames, height, width, dim = tokens.shape
    if height < 1 or width < 1:
        raise ValueError(f"Token grid must be non-empty, got {height}x{width}")

    # Compute interpolation in float32 even when the model runs in bfloat16.
    grid = tokens.astype(jnp.float32).reshape(batch_size, num_frames, height * width, dim)
    points = jnp.clip(coordinates.astype(jnp.float32), 0.0, 1.0)
    x = points[..., 0] * max(width - 1, 0)
    y = points[..., 1] * max(height - 1, 0)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y1 = jnp.minimum(y0 + 1, height - 1)
    dx = x - x0.astype(x.dtype)
    dy = y - y0.astype(y.dtype)

    def gather(row, column):
        indices = row * width + column
        return jnp.take_along_axis(grid, indices[..., None], axis=2)

    top_left = gather(y0, x0)
    top_right = gather(y0, x1)
    bottom_left = gather(y1, x0)
    bottom_right = gather(y1, x1)
    weights = (
        (1.0 - dx) * (1.0 - dy),
        dx * (1.0 - dy),
        (1.0 - dx) * dy,
        dx * dy,
    )
    sampled = (
        top_left * weights[0][..., None]
        + top_right * weights[1][..., None]
        + bottom_left * weights[2][..., None]
        + bottom_right * weights[3][..., None]
    )
    # Keep the interpolation in float32 until the caller combines it with
    # coordinate embeddings. The final cast happens once before prefix
    # construction, avoiding an unnecessary low-precision round-trip here.
    return sampled


def masked_global_average_image_tokens(tokens, frame_mask):
    """Average each spatial token grid while zeroing invalid support frames."""
    if tokens.ndim != 5 or frame_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"Expected tokens [B,K,H,W,D] and frame_mask [B,K], got {tokens.shape} and {frame_mask.shape}"
        )
    valid = frame_mask.astype(jnp.bool_)[..., None, None, None]
    finite_tokens = jnp.where(valid, tokens.astype(jnp.float32), 0.0)
    return jnp.mean(finite_tokens, axis=(2, 3))


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.use_support_context = config.use_support_context
        self.use_support_progress = config.use_support_progress
        self.use_support_role_embedding = config.use_support_role_embedding
        self.support_progress_embed_dim = config.support_progress_embed_dim
        self.use_support_token_compression = config.use_support_token_compression
        self.support_static_tokens = config.support_static_tokens
        self.support_motion_tokens_per_frame = config.support_motion_tokens_per_frame
        self.support_compression_temperature = config.support_compression_temperature
        self.use_support_grid_sampling = config.use_support_grid_sampling
        self.support_grid_hidden_dim = config.support_grid_hidden_dim
        self.support_grid_tokens_per_frame = config.support_grid_tokens_per_frame
        self.use_caption_supervision = config.use_caption_supervision
        self.caption_decode_chunk_size = config.caption_decode_chunk_size
        self.caption_loss_weight = config.caption_loss_weight
        self.num_caption_queries = config.num_caption_queries
        self.caption_robot_tokens_per_image = config.caption_robot_tokens_per_image

        if self.use_support_context and self.use_support_progress:
            self.support_progress_mlp_in = nnx.Linear(
                2 * self.support_progress_embed_dim,
                paligemma_config.width,
                rngs=rngs,
            )
            self.support_progress_mlp_out = nnx.Linear(
                paligemma_config.width,
                paligemma_config.width,
                rngs=rngs,
            )
        if self.use_support_context and self.use_support_grid_sampling:
            # GridS predicts S continuous (x, y) points from each frame's
            # global visual summary. Parameters are shared by all frames.
            self.support_grid_mlp_in = nnx.Linear(
                paligemma_config.width,
                self.support_grid_hidden_dim,
                rngs=rngs,
            )
            self.support_grid_mlp_out = nnx.Linear(
                self.support_grid_hidden_dim,
                2 * self.support_grid_tokens_per_frame,
                rngs=rngs,
            )
            self.support_grid_coord_proj = nnx.Linear(
                2,
                paligemma_config.width,
                rngs=rngs,
            )
        if self.use_support_context and self.use_support_role_embedding:
            # 0=robot image, 1=support image, 2=caption, 3=task prompt,
            # 4=caption semantic query.
            role_count = 5 if self.use_caption_supervision else 4
            self.support_role_embeddings = nnx.Param(
                jnp.zeros((role_count, paligemma_config.width), dtype=jnp.dtype(config.dtype))
            )
        if self.use_caption_supervision:
            # Query tokens are the only caption-side states that are exposed to
            # the action branch. Distinct small random values break symmetry
            # between queries; the small gate limits their initial influence.
            self.caption_query_tokens = nnx.Param(
                (
                    0.02
                    * jax.random.normal(
                        rngs.params(),
                        (self.num_caption_queries, paligemma_config.width),
                    )
                ).astype(jnp.dtype(config.dtype))
            )
            self.caption_action_gate = nnx.Param(
                jnp.asarray(config.caption_action_gate_init, dtype=jnp.dtype(config.dtype))
            )
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _role_embedding(self, role_index: int, dtype):
        if not self.use_support_context or not self.use_support_role_embedding:
            return None
        return self.support_role_embeddings.value[role_index].astype(dtype)

    def _caption_query_inputs(self, context_token_mask, dtype):
        """Return learned semantic query tokens for the caption branch."""
        batch_size = context_token_mask.shape[0]
        query_tokens = jnp.broadcast_to(
            self.caption_query_tokens.value.astype(dtype),
            (batch_size, self.num_caption_queries, self.caption_query_tokens.value.shape[-1]),
        )
        role = self._role_embedding(4, dtype)
        if role is not None:
            query_tokens = query_tokens + role
        # Keep queries valid even when visual context is masked. This gives the
        # caption branch a finite attention row during ablations.
        query_mask = jnp.ones(
            (batch_size, self.num_caption_queries),
            dtype=jnp.bool_,
        )
        return query_tokens, query_mask

    def _support_progress_embedding(
        self,
        support_frame_progress,
        chunk_progress,
        tokens_per_frame: int,
        dtype,
    ):
        batch_size, num_frames = support_frame_progress.shape
        frame_embedding = posemb_sincos(
            support_frame_progress.astype(jnp.float32).reshape(-1),
            self.support_progress_embed_dim,
            min_period=4e-3,
            max_period=4.0,
        ).reshape(batch_size, num_frames, -1)

        if chunk_progress is None:
            chunk_embedding = jnp.zeros_like(frame_embedding)
        else:
            chunk_progress = jnp.broadcast_to(chunk_progress.astype(jnp.float32), (batch_size, num_frames))
            chunk_embedding = posemb_sincos(
                chunk_progress.reshape(-1),
                self.support_progress_embed_dim,
                min_period=4e-3,
                max_period=4.0,
            ).reshape(batch_size, num_frames, -1)

        progress_embedding = jnp.concatenate([frame_embedding, chunk_embedding], axis=-1)
        progress_embedding = self.support_progress_mlp_in(progress_embedding)
        progress_embedding = nnx.swish(progress_embedding)
        progress_embedding = self.support_progress_mlp_out(progress_embedding)
        progress_embedding = progress_embedding.astype(dtype)
        return einops.repeat(progress_embedding, "b k d -> b k s d", s=tokens_per_frame)

    def _encode_support_images(self, obs: _model.Observation):
        if obs.support_images is None:
            raise ValueError("use_support_context=True requires support_images")
        batch_size, num_frames = obs.support_images.shape[:2]
        flat_images = einops.rearrange(obs.support_images, "b k h w c -> (b k) h w c")
        image_tokens, _ = self.PaliGemma.img(flat_images, train=False)
        return einops.rearrange(
            image_tokens,
            "(b k) s d -> b k s d",
            b=batch_size,
            k=num_frames,
        )

    def _encode_robot_images(self, obs: _model.Observation):
        return {name: self.PaliGemma.img(image, train=False)[0] for name, image in obs.images.items()}

    def _grid_sample_support_tokens(self, support_image_tokens, support_image_mask):
        """Apply a shared, frame-wise GridS sampler to support SigLIP tokens."""
        batch_size, num_frames, tokens_per_frame, dim = support_image_tokens.shape
        grid_size = int(tokens_per_frame**0.5)
        if grid_size * grid_size != tokens_per_frame:
            raise ValueError(
                f"Support image tokens must form a square grid for GridS, got {tokens_per_frame}"
            )

        grid = einops.rearrange(
            support_image_tokens.astype(jnp.float32),
            "b k (h w) d -> b k h w d",
            h=grid_size,
            w=grid_size,
        )
        pooled = masked_global_average_image_tokens(grid, support_image_mask)
        pooled = pooled.reshape(batch_size * num_frames, dim)
        hidden = self.support_grid_mlp_in(pooled)
        hidden = nnx.swish(hidden)
        coord_logits = self.support_grid_mlp_out(hidden)
        coordinates = jax.nn.sigmoid(
            coord_logits.reshape(batch_size, num_frames, self.support_grid_tokens_per_frame, 2)
        )

        sampled = bilinear_sample_image_tokens(grid, coordinates)

        # Inject the predicted coordinates after sampling, as in GridS. This
        # preserves spatial identity even when two points read similar visual
        # features. The projection is shared across all frames and points.
        coord_features = self.support_grid_coord_proj(coordinates * 2.0 - 1.0)
        sampled = sampled.astype(coord_features.dtype) + coord_features
        sampled = sampled * support_image_mask.astype(sampled.dtype)[..., None, None]
        return sampled.astype(support_image_tokens.dtype)

    def _prepare_robot_tokens(self, obs: _model.Observation, robot_image_tokens):
        tokens = []
        masks = []
        for name in obs.images:
            image_tokens = robot_image_tokens[name]
            role = self._role_embedding(0, image_tokens.dtype)
            if role is not None:
                image_tokens = image_tokens + role
            tokens.append(image_tokens)
            masks.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1)

    def _prepare_support_tokens(
        self,
        obs: _model.Observation,
        support_image_tokens,
        *,
        include_chunk_progress: bool,
    ):
        batch_size, num_frames, tokens_per_frame, _ = support_image_tokens.shape
        support_image_mask = obs.support_image_mask
        if support_image_mask is None:
            support_image_mask = jnp.ones((batch_size, num_frames), dtype=jnp.bool_)
        else:
            support_image_mask = support_image_mask.astype(jnp.bool_)

        # GridS operates on raw SigLIP features. Progress is deliberately
        # attached only after sampling so that phase metadata cannot change the
        # learned visual coordinates (and caption context never sees robot
        # chunk progress).
        if self.use_support_grid_sampling:
            support_tokens = self._grid_sample_support_tokens(support_image_tokens, support_image_mask)
            tokens_per_frame = self.support_grid_tokens_per_frame
            value_tokens = support_tokens
        else:
            value_tokens = support_image_tokens
        if self.use_support_progress:
            if obs.support_frame_progress is None:
                raise ValueError("use_support_progress=True requires support_frame_progress")
            if include_chunk_progress and obs.chunk_progress is None:
                raise ValueError("Action support context requires chunk_progress")
            progress_embedding = self._support_progress_embedding(
                obs.support_frame_progress,
                obs.chunk_progress if include_chunk_progress else None,
                tokens_per_frame,
                support_image_tokens.dtype,
            )
            value_tokens = value_tokens + progress_embedding

        if self.use_support_grid_sampling:
            support_tokens = einops.rearrange(value_tokens, "b k s d -> b (k s) d")
            support_token_mask = einops.repeat(
                support_image_mask,
                "b k -> b (k s)",
                s=tokens_per_frame,
            )
        elif self.use_support_token_compression:
            support_tokens, support_token_mask = compress_support_image_tokens(
                value_tokens,
                support_image_tokens,
                support_image_mask,
                num_static_tokens=self.support_static_tokens,
                motion_tokens_per_frame=self.support_motion_tokens_per_frame,
                temperature=self.support_compression_temperature,
            )
        else:
            support_tokens = einops.rearrange(value_tokens, "b k s d -> b (k s) d")
            support_token_mask = einops.repeat(
                support_image_mask,
                "b k -> b (k s)",
                s=tokens_per_frame,
            )

        role = self._role_embedding(1, support_tokens.dtype)
        if role is not None:
            support_tokens = support_tokens + role
        return support_tokens, support_token_mask

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
        support_image_tokens: at.Float[at.Array, "b k support_s d"] | None = None,
        caption_semantic_tokens: at.Float[at.Array, "b query_s d"] | None = None,
        caption_semantic_mask: at.Bool[at.Array, "b query_s"] | None = None,
        robot_image_tokens=None,
    ) -> tuple[
        at.Float[at.Array, "b prefix_s emb"],
        at.Bool[at.Array, "b prefix_s"],
        at.Bool[at.Array, " prefix_s"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # Robot observation images.
        if robot_image_tokens is None:
            robot_image_tokens = self._encode_robot_images(obs)
        robot_tokens, robot_token_mask = self._prepare_robot_tokens(obs, robot_image_tokens)
        tokens.append(robot_tokens)
        input_mask.append(robot_token_mask)
        # image tokens attend to each other
        ar_mask += [False] * robot_tokens.shape[1]

        # Human support video. Caption tokens are intentionally not part of the
        # action prefix because captions are unavailable at inference time.
        if self.use_support_context:
            if support_image_tokens is None:
                support_image_tokens = self._encode_support_images(obs)
            # Robot trajectory progress is only a training-time label source for
            # phase-caption selection. It is not a model input because no
            # semantically equivalent completion progress exists at inference.
            support_tokens, support_token_mask = self._prepare_support_tokens(
                obs,
                support_image_tokens,
                include_chunk_progress=False,
            )
            tokens.append(support_tokens)
            input_mask.append(support_token_mask)
            ar_mask += [False] * support_tokens.shape[1]

        if caption_semantic_tokens is not None:
            if caption_semantic_mask is None:
                raise ValueError("caption_semantic_tokens requires caption_semantic_mask")
            gate = jnp.tanh(self.caption_action_gate.value).astype(caption_semantic_tokens.dtype)
            tokens.append(caption_semantic_tokens * gate)
            input_mask.append(caption_semantic_mask)
            ar_mask += [False] * caption_semantic_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")

            role = self._role_embedding(3, tokenized_inputs.dtype)
            if role is not None:
                tokenized_inputs = tokenized_inputs + role

            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _prepare_caption_context(
        self,
        obs: _model.Observation,
        support_image_tokens,
        robot_image_tokens,
    ):
        pooled_robot_image_tokens = {
            name: spatial_pool_image_tokens(
                image_tokens,
                num_tokens=self.caption_robot_tokens_per_image,
            )
            for name, image_tokens in robot_image_tokens.items()
        }
        robot_tokens, robot_token_mask = self._prepare_robot_tokens(obs, pooled_robot_image_tokens)
        support_tokens, support_token_mask = self._prepare_support_tokens(
            obs,
            support_image_tokens,
            include_chunk_progress=False,
        )
        return (
            jnp.concatenate([robot_tokens, support_tokens], axis=1),
            jnp.concatenate([robot_token_mask, support_token_mask], axis=1),
        )

    @at.typecheck
    def embed_caption_prefix(
        self,
        obs: _model.Observation,
        support_image_tokens: at.Float[at.Array, "b k support_s d"] | None = None,
        robot_image_tokens=None,
    ) -> tuple[
        at.Float[at.Array, "b caption_prefix_s emb"],
        at.Bool[at.Array, "b caption_prefix_s"],
        at.Bool[at.Array, "b caption_prefix_s caption_prefix_s"],
    ]:
        """Build the robot-and-video-to-caption teacher-forcing sequence.

        Current robot and support-video tokens condition semantic queries without
        receiving numeric chunk progress. Caption inputs are causal, cannot be
        read by the queries, and access visual information only through queries.
        """
        if not self.use_caption_supervision:
            raise ValueError("Caption supervision is disabled in the model config")
        if obs.caption_input_tokens is None or obs.caption_input_mask is None:
            raise ValueError("Caption supervision requires caption_input_tokens and caption_input_mask")
        if support_image_tokens is None:
            support_image_tokens = self._encode_support_images(obs)
        if robot_image_tokens is None:
            robot_image_tokens = self._encode_robot_images(obs)

        context_tokens, context_mask = self._prepare_caption_context(
            obs,
            support_image_tokens,
            robot_image_tokens,
        )
        query_tokens, query_mask = self._caption_query_inputs(
            context_mask,
            context_tokens.dtype,
        )
        caption_tokens = self.PaliGemma.llm(obs.caption_input_tokens, method="embed")
        role = self._role_embedding(2, caption_tokens.dtype)
        if role is not None:
            caption_tokens = caption_tokens + role

        # A fully masked row can make attention normalization undefined. The
        # loss mask remains unchanged, so this pad embedding is never supervised.
        caption_input_mask = ensure_nonempty_caption_input_mask(
            jnp.concatenate([context_mask, query_mask], axis=1),
            obs.caption_input_mask,
        )

        tokens = jnp.concatenate([context_tokens, query_tokens, caption_tokens], axis=1)
        input_mask = jnp.concatenate([context_mask, query_mask, caption_input_mask], axis=1)
        attn_mask = make_caption_bottleneck_attn_mask(
            input_mask,
            context_len=context_tokens.shape[1],
            query_len=query_tokens.shape[1],
        )
        return tokens, input_mask, attn_mask

    def compute_caption_outputs(
        self,
        observation: _model.Observation,
        support_image_tokens: at.Float[at.Array, "b k s d"] | None = None,
        robot_image_tokens=None,
    ):
        """Return per-sample caption metrics and phase-aware semantic tokens."""
        caption_fields = {
            "caption_input_tokens": observation.caption_input_tokens,
            "caption_input_mask": observation.caption_input_mask,
            "caption_target_tokens": observation.caption_target_tokens,
            "caption_loss_mask": observation.caption_loss_mask,
            "caption_hand_side_mask": observation.caption_hand_side_mask,
        }
        missing_fields = [name for name, value in caption_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Caption supervision is missing fields: {missing_fields}")

        prefix_tokens, prefix_mask, attn_mask = self.embed_caption_prefix(
            observation,
            support_image_tokens=support_image_tokens,
            robot_image_tokens=robot_image_tokens,
        )
        positions = jnp.maximum(jnp.cumsum(prefix_mask, axis=1) - 1, 0)
        (prefix_out, action_expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attn_mask,
            positions=positions,
            deterministic=self.deterministic,
        )
        assert prefix_out is not None
        assert action_expert_out is None

        caption_len = observation.caption_input_tokens.shape[1]
        query_start = prefix_out.shape[1] - caption_len - self.num_caption_queries
        caption_semantic_tokens = prefix_out[:, query_start : query_start + self.num_caption_queries]
        caption_semantic_mask = prefix_mask[:, query_start : query_start + self.num_caption_queries]
        caption_hidden = prefix_out[:, -caption_len:]
        caption_metrics = compute_caption_token_metrics(
            caption_hidden,
            observation.caption_target_tokens,
            observation.caption_loss_mask,
            observation.caption_hand_side_mask,
            decode_fn=lambda hidden: self.PaliGemma.llm(hidden, method="decode"),
            chunk_size=self.caption_decode_chunk_size,
        )
        return caption_metrics, caption_semantic_tokens, caption_semantic_mask

    def compute_caption_metrics(
        self,
        observation: _model.Observation,
        support_image_tokens: at.Float[at.Array, "b k s d"] | None = None,
        robot_image_tokens=None,
    ):
        """Run the caption branch while preserving the original metrics API."""
        metrics, _, _ = self.compute_caption_outputs(
            observation,
            support_image_tokens=support_image_tokens,
            robot_image_tokens=robot_image_tokens,
        )
        return metrics

    def compute_caption_semantic_tokens(
        self,
        observation: _model.Observation,
        support_image_tokens: at.Float[at.Array, "b k s d"] | None = None,
        robot_image_tokens=None,
    ):
        """Extract Z_cap from current robot images and support video."""
        if not self.use_caption_supervision:
            raise ValueError("Caption semantic tokens require use_caption_supervision=True")
        if support_image_tokens is None:
            support_image_tokens = self._encode_support_images(observation)
        if robot_image_tokens is None:
            robot_image_tokens = self._encode_robot_images(observation)
        context_tokens, context_mask = self._prepare_caption_context(
            observation,
            support_image_tokens,
            robot_image_tokens,
        )
        query_tokens, query_mask = self._caption_query_inputs(
            context_mask,
            context_tokens.dtype,
        )
        prefix_tokens = jnp.concatenate([context_tokens, query_tokens], axis=1)
        prefix_mask = jnp.concatenate([context_mask, query_mask], axis=1)
        prefix_ar_mask = jnp.zeros((prefix_tokens.shape[1],), dtype=jnp.bool_)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.maximum(jnp.cumsum(prefix_mask, axis=1) - 1, 0)
        (prefix_out, action_expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attn_mask,
            positions=positions,
            deterministic=self.deterministic,
        )
        assert prefix_out is not None
        assert action_expert_out is None
        return prefix_out[:, -self.num_caption_queries :], query_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_action_loss(
        self,
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        support_image_tokens: at.Float[at.Array, "b k support_s d"] | None,
        robot_image_tokens=None,
        caption_semantic_tokens: at.Float[at.Array, "b query_s d"] | None = None,
        caption_semantic_mask: at.Bool[at.Array, "b query_s"] | None = None,
    ):
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation,
            support_image_tokens=support_image_tokens,
            caption_semantic_tokens=caption_semantic_tokens,
            caption_semantic_mask=caption_semantic_mask,
            robot_image_tokens=robot_image_tokens,
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_metrics(rng, observation, actions, train=train)
        return loss

    @override
    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        robot_image_tokens = self._encode_robot_images(observation)
        support_image_tokens = self._encode_support_images(observation) if self.use_support_context else None
        caption_metrics = None
        caption_semantic_tokens = None
        caption_semantic_mask = None
        if self.use_caption_supervision:
            per_sample_caption_metrics, caption_semantic_tokens, caption_semantic_mask = self.compute_caption_outputs(
                observation,
                support_image_tokens=support_image_tokens,
                robot_image_tokens=robot_image_tokens,
            )
            caption_metrics = aggregate_caption_metrics(per_sample_caption_metrics)

        action_loss = self._compute_action_loss(
            noise_rng,
            time_rng,
            observation,
            actions,
            support_image_tokens,
            robot_image_tokens,
            caption_semantic_tokens,
            caption_semantic_mask,
        )
        metrics = {"action_loss": jnp.mean(action_loss)}
        if not self.use_caption_supervision:
            return action_loss, metrics

        assert caption_metrics is not None
        weighted_caption_loss = self.caption_loss_weight * caption_metrics["caption_loss"]
        metrics.update(caption_metrics)
        metrics["caption_weighted_loss"] = weighted_caption_loss
        metrics["caption_action_gate"] = jnp.tanh(self.caption_action_gate.value).astype(jnp.float32)
        return action_loss + weighted_caption_loss, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        robot_image_tokens = self._encode_robot_images(observation)
        support_image_tokens = self._encode_support_images(observation) if self.use_support_context else None
        caption_semantic_tokens = None
        caption_semantic_mask = None
        if self.use_caption_supervision:
            caption_semantic_tokens, caption_semantic_mask = self.compute_caption_semantic_tokens(
                observation,
                support_image_tokens=support_image_tokens,
                robot_image_tokens=robot_image_tokens,
            )

        # First fill KV cache with a forward pass of the action prefix.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation,
            support_image_tokens=support_image_tokens,
            caption_semantic_tokens=caption_semantic_tokens,
            caption_semantic_mask=caption_semantic_mask,
            robot_image_tokens=robot_image_tokens,
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
