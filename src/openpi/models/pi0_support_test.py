import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config


def test_compress_support_image_tokens_shape_and_mask():
    tokens = jnp.arange(2 * 3 * 16 * 4, dtype=jnp.float32).reshape(2, 3, 16, 4)
    frame_mask = jnp.asarray([[True, True, True], [True, False, False]])

    compressed, mask = pi0.compress_support_image_tokens(
        tokens,
        tokens,
        frame_mask,
        num_static_tokens=4,
        motion_tokens_per_frame=2,
        temperature=1.0,
    )

    assert compressed.shape == (2, 10, 4)
    assert mask.shape == (2, 10)
    assert mask[0].all()
    assert mask[1, :6].all()
    assert not mask[1, 6:].any()


def test_motion_selection_uses_raw_visual_scores():
    score_tokens = np.zeros((1, 2, 4, 1), dtype=np.float32)
    score_tokens[0, 1, 3, 0] = 5.0
    value_tokens = np.zeros_like(score_tokens)
    value_tokens[0, 0, :, 0] = np.arange(4)
    value_tokens[0, 1, :, 0] = np.arange(10, 14)

    compressed, _ = pi0.compress_support_image_tokens(
        jnp.asarray(value_tokens),
        jnp.asarray(score_tokens),
        jnp.ones((1, 2), dtype=jnp.bool_),
        num_static_tokens=1,
        motion_tokens_per_frame=1,
        temperature=1.0,
    )

    # The final two tokens are the selected motion patch from each frame.
    assert np.array_equal(np.asarray(compressed[0, 1:, 0]), np.asarray([3.0, 13.0]))


def test_compress_support_image_tokens_masks_null_support():
    tokens = jnp.ones((1, 2, 4, 2), dtype=jnp.float32)
    _, mask = pi0.compress_support_image_tokens(
        tokens,
        tokens,
        jnp.zeros((1, 2), dtype=jnp.bool_),
        num_static_tokens=1,
        motion_tokens_per_frame=1,
        temperature=1.0,
    )

    assert not mask.any()


def test_spatial_pool_image_tokens_preserves_local_grid_layout():
    tokens = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)

    pooled = pi0.spatial_pool_image_tokens(tokens, num_tokens=4)

    assert pooled.shape == (1, 4, 1)
    np.testing.assert_allclose(
        np.asarray(pooled[0, :, 0]),
        np.asarray([2.5, 4.5, 10.5, 12.5]),
    )


def test_support_input_spec():
    config = pi0_config.Pi0Config(use_support_context=True, num_support_frames=8)
    observation, _ = config.inputs_spec(batch_size=2)

    assert observation.support_images.shape == (2, 8, 224, 224, 3)
    assert observation.support_image_mask.shape == (2, 8)
    assert observation.support_frame_progress.shape == (2, 8)
    assert observation.chunk_progress.shape == (2, 1)


def test_support_config_rejects_invalid_compression():
    with pytest.raises(ValueError, match="square number"):
        pi0_config.Pi0Config(support_static_tokens=63)


def test_support_model_abstract_init():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_support_context=True,
        support_progress_embed_dim=8,
        support_static_tokens=4,
        support_motion_tokens_per_frame=2,
    )

    model = nnx.eval_shape(config.create, jax.random.key(0))

    assert model.support_role_embeddings.value.shape == (4, 64)
    assert model.support_progress_mlp_in.in_features == 16
    assert model.support_progress_mlp_out.out_features == 64


def test_caption_semantic_parameters_are_initialized():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_support_context=True,
        use_caption_supervision=True,
        num_caption_queries=3,
        support_progress_embed_dim=8,
        support_static_tokens=4,
        support_motion_tokens_per_frame=2,
    )

    model = nnx.eval_shape(config.create, jax.random.key(0))

    assert model.support_role_embeddings.value.shape == (5, 64)
    assert model.caption_query_tokens.value.shape == (3, 64)
    assert model.caption_action_gate.value.shape == ()
