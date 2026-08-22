import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0
from openpi.models import pi0_config


def test_caption_prefix_attention_is_video_prefix_plus_causal_caption():
    input_mask = jnp.ones((1, 5), dtype=jnp.bool_)
    # Two bidirectional video tokens followed by three causal caption tokens.
    ar_mask = jnp.asarray([False, False, True, True, True])

    attention = pi0.make_attn_mask(input_mask, ar_mask)

    expected = np.asarray(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    np.testing.assert_array_equal(np.asarray(attention[0]), expected)


def test_caption_bottleneck_isolates_queries_and_raw_visual_context():
    input_mask = jnp.ones((1, 7), dtype=jnp.bool_)
    # Two raw visual tokens, two semantic queries, then three caption tokens.
    attention = np.asarray(
        pi0.make_caption_bottleneck_attn_mask(
            input_mask,
            context_len=2,
            query_len=2,
        )[0]
    )

    # Visual and query tokens share a bidirectional prefix and cannot read truth.
    assert attention[:4, :4].all()
    assert not attention[:4, 4:].any()

    # Caption tokens read queries and causal caption history, never raw visuals.
    assert not attention[4:, :2].any()
    assert attention[4:, 2:4].all()
    assert attention[4, 4]
    assert not attention[4, 5:].any()
    assert attention[5, 4:6].all()
    assert not attention[5, 6]
    assert attention[6, 4:7].all()


def test_caption_prefix_adds_inert_position_only_for_fully_masked_rows():
    support_mask = jnp.asarray([[True, False], [False, False], [False, False]])
    caption_mask = jnp.asarray(
        [
            [False, False, False],
            [True, True, False],
            [False, False, False],
        ]
    )

    safe_mask = pi0.ensure_nonempty_caption_input_mask(support_mask, caption_mask)

    np.testing.assert_array_equal(
        np.asarray(safe_mask),
        np.asarray(
            [
                [False, False, False],
                [True, True, False],
                [True, False, False],
            ]
        ),
    )


def test_chunked_caption_metrics_match_full_cross_entropy():
    logits = jnp.asarray(
        [
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
                [8.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    targets = jnp.asarray([[0, 1, 2, 2, 3], [0, 1, 2, 3, 0]], dtype=jnp.int32)
    loss_mask = jnp.asarray([[True, True, False, True, False], [False, False, False, False, False]])
    hand_mask = jnp.asarray([[False, True, True, False, False], [False, False, False, False, False]])
    decoded_chunk_sizes = []

    def decode_fn(hidden):
        decoded_chunk_sizes.append(hidden.shape[1])
        return hidden

    metrics = pi0.compute_caption_token_metrics(
        logits,
        targets,
        loss_mask,
        hand_mask,
        decode_fn=decode_fn,
        chunk_size=2,
    )

    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    full_nll = jax.nn.logsumexp(logits, axis=-1) - target_logits
    expected_loss = jnp.sum(full_nll[0] * loss_mask[0]) / 3
    np.testing.assert_allclose(metrics["caption_loss"], jnp.asarray([expected_loss, 0.0]), rtol=1e-6)
    np.testing.assert_allclose(metrics["caption_token_accuracy"], jnp.asarray([2 / 3, 0.0]), rtol=1e-6)
    np.testing.assert_allclose(metrics["caption_hand_side_loss"], jnp.asarray([full_nll[0, 1], 0.0]), rtol=1e-6)
    np.testing.assert_allclose(metrics["caption_hand_side_accuracy"], jnp.asarray([1.0, 0.0]), rtol=1e-6)
    np.testing.assert_array_equal(metrics["caption_token_count"], jnp.asarray([3, 0]))
    np.testing.assert_array_equal(metrics["caption_hand_side_token_count"], jnp.asarray([1, 0]))
    assert decoded_chunk_sizes == [2, 2, 1]


def test_joint_loss_shares_support_encoding_and_uses_token_weighted_caption_loss():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=2,
        use_support_context=True,
        use_caption_supervision=True,
        num_support_frames=1,
        caption_max_len=3,
        caption_loss_weight=0.1,
    )
    observation = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)

    class JointLossHarness:
        use_support_context = True
        use_caption_supervision = True
        caption_loss_weight = 0.1

        def __init__(self):
            self.support_encode_calls = 0
            self.robot_encode_calls = 0
            self.caption_forward_calls = 0
            self.action_forward_calls = 0
            self.encoded_support = jnp.ones((2, 1, 1, 1), dtype=jnp.float32)
            self.encoded_robot = {"base_0_rgb": jnp.ones((2, 1, 1), dtype=jnp.float32)}

        def _encode_support_images(self, obs):
            del obs
            self.support_encode_calls += 1
            return self.encoded_support

        def _encode_robot_images(self, obs):
            del obs
            self.robot_encode_calls += 1
            return self.encoded_robot

        def _compute_action_loss(
            self,
            noise_rng,
            time_rng,
            obs,
            action_targets,
            support_image_tokens,
            robot_image_tokens,
            caption_semantic_tokens,
            caption_semantic_mask,
        ):
            del noise_rng, time_rng, obs, action_targets
            self.action_forward_calls += 1
            assert support_image_tokens is self.encoded_support
            assert robot_image_tokens is self.encoded_robot
            assert caption_semantic_tokens.shape == (2, 2, 1)
            assert caption_semantic_mask.shape == (2, 2)
            return jnp.asarray([[1.0, 3.0], [5.0, 7.0]])

        def compute_caption_outputs(self, obs, support_image_tokens, robot_image_tokens):
            del obs
            self.caption_forward_calls += 1
            assert support_image_tokens is self.encoded_support
            assert robot_image_tokens is self.encoded_robot
            metrics = {
                "caption_loss": jnp.asarray([2.0, 4.0]),
                "caption_token_accuracy": jnp.asarray([1.0, 0.5]),
                "caption_hand_side_loss": jnp.asarray([0.0, 6.0]),
                "caption_hand_side_accuracy": jnp.asarray([0.0, 1.0]),
                "caption_token_count": jnp.asarray([1, 3]),
                "caption_hand_side_token_count": jnp.asarray([0, 1]),
            }
            semantic_tokens = jnp.ones((2, 2, 1), dtype=jnp.float32)
            semantic_mask = jnp.ones((2, 2), dtype=jnp.bool_)
            return metrics, semantic_tokens, semantic_mask

        class _Gate:
            value = jnp.asarray(0.1)

        caption_action_gate = _Gate()

    harness = JointLossHarness()
    loss, metrics = pi0.Pi0.compute_loss_with_metrics(
        harness,
        jax.random.key(0),
        observation,
        actions,
    )

    # Caption CE is (2 * 1 + 4 * 3) / 4 = 3.5, then weighted by 0.1.
    np.testing.assert_allclose(loss, jnp.asarray([[1.35, 3.35], [5.35, 7.35]]), rtol=1e-6)
    np.testing.assert_allclose(metrics["action_loss"], 4.0)
    np.testing.assert_allclose(metrics["caption_loss"], 3.5)
    np.testing.assert_allclose(metrics["caption_weighted_loss"], 0.35)
    np.testing.assert_allclose(metrics["caption_token_accuracy"], 0.625)
    assert metrics["caption_token_count"] == 4
    assert metrics["caption_valid_samples"] == 2
    assert harness.support_encode_calls == 1
    assert harness.robot_encode_calls == 1
    assert harness.caption_forward_calls == 1
    assert harness.action_forward_calls == 1


def test_dummy_caption_forward_shape_includes_robot_context():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_support_context=True,
        use_caption_supervision=True,
        num_support_frames=1,
        caption_max_len=4,
        caption_decode_chunk_size=2,
        support_progress_embed_dim=8,
        support_static_tokens=4,
        support_motion_tokens_per_frame=1,
        num_caption_queries=3,
    )
    observation = config.fake_obs()
    support_image_tokens = jnp.ones((1, 1, 256, 64), dtype=jnp.float32)

    def create_and_forward(key, obs, encoded_support):
        model = config.create(key)
        encoded_robot = model._encode_robot_images(obs)  # noqa: SLF001
        caption_prefix = model.embed_caption_prefix(obs, encoded_support, encoded_robot)
        metrics, semantic_tokens, semantic_mask = model.compute_caption_outputs(
            obs,
            encoded_support,
            encoded_robot,
        )
        return caption_prefix, metrics, semantic_tokens, semantic_mask

    (prefix_tokens, prefix_mask, prefix_attn_mask), metrics, semantic_tokens, semantic_mask = jax.eval_shape(
        create_and_forward,
        jax.random.key(0),
        observation,
        support_image_tokens,
    )

    # 3 x 64 pooled robot tokens + 4 static + 1 motion support token + 3 queries
    # + 4 caption positions. Instruction, state, action, and chunk progress are absent.
    assert prefix_tokens.shape == (1, 204, 64)
    assert prefix_mask.shape == (1, 204)
    assert prefix_attn_mask.shape == (1, 204, 204)
    assert semantic_tokens.shape == (1, 3, 64)
    assert semantic_mask.shape == (1, 3)
    assert metrics["caption_loss"].shape == (1,)
    assert metrics["caption_token_count"].dtype == jnp.int32

    inference_observation = observation.replace(
        caption_input_tokens=None,
        caption_input_mask=None,
        caption_target_tokens=None,
        caption_loss_mask=None,
        caption_hand_side_mask=None,
    )

    def create_and_extract(key, obs, encoded_support):
        model = config.create(key)
        encoded_robot = model._encode_robot_images(obs)  # noqa: SLF001
        return model.compute_caption_semantic_tokens(obs, encoded_support, encoded_robot)

    inference_tokens, inference_mask = jax.eval_shape(
        create_and_extract,
        jax.random.key(1),
        inference_observation,
        support_image_tokens,
    )
    assert inference_tokens.shape == (1, 3, 64)
    assert inference_mask.shape == (1, 3)


def test_semantic_tokens_match_without_caption_truth_or_chunk_progress():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_support_context=True,
        use_caption_supervision=True,
        num_support_frames=1,
        caption_max_len=4,
        caption_decode_chunk_size=2,
        support_progress_embed_dim=8,
        support_static_tokens=4,
        support_motion_tokens_per_frame=1,
        num_caption_queries=2,
        caption_robot_tokens_per_image=1,
    )
    model = config.create(jax.random.key(7))
    observation = config.fake_obs().replace(
        caption_input_tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        caption_input_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        caption_target_tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        caption_loss_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        caption_hand_side_mask=jnp.zeros((1, 4), dtype=jnp.bool_),
        chunk_progress=jnp.zeros((1, 1), dtype=jnp.float32),
    )
    support_tokens = jnp.ones((1, 1, 256, 64), dtype=jnp.float32)
    robot_tokens = {
        name: jnp.full((1, 4, 64), index + 1, dtype=jnp.float32)
        for index, name in enumerate(observation.images)
    }

    _, training_tokens_a, _ = model.compute_caption_outputs(
        observation,
        support_image_tokens=support_tokens,
        robot_image_tokens=robot_tokens,
    )
    changed_supervision = observation.replace(
        caption_input_tokens=jnp.full((1, 4), 3, dtype=jnp.int32),
        caption_target_tokens=jnp.full((1, 4), 4, dtype=jnp.int32),
        chunk_progress=jnp.ones((1, 1), dtype=jnp.float32),
    )
    _, training_tokens_b, _ = model.compute_caption_outputs(
        changed_supervision,
        support_image_tokens=support_tokens,
        robot_image_tokens=robot_tokens,
    )
    inference_observation = changed_supervision.replace(
        caption_input_tokens=None,
        caption_input_mask=None,
        caption_target_tokens=None,
        caption_loss_mask=None,
        caption_hand_side_mask=None,
    )
    inference_tokens, _ = model.compute_caption_semantic_tokens(
        inference_observation,
        support_image_tokens=support_tokens,
        robot_image_tokens=robot_tokens,
    )

    np.testing.assert_allclose(
        np.asarray(training_tokens_a, dtype=np.float32),
        np.asarray(training_tokens_b, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(training_tokens_a, dtype=np.float32),
        np.asarray(inference_tokens, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dummy_joint_caption_and_action_loss_shape():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        use_support_context=True,
        use_caption_supervision=True,
        num_support_frames=1,
        caption_max_len=4,
        caption_decode_chunk_size=2,
        support_progress_embed_dim=8,
        support_static_tokens=4,
        support_motion_tokens_per_frame=1,
        num_caption_queries=2,
        caption_robot_tokens_per_image=1,
    )
    observation = config.fake_obs()
    actions = config.fake_act()

    def create_and_compute_loss(key, obs, action_targets):
        model = config.create(key)
        return model.compute_loss_with_metrics(key, obs, action_targets)

    loss, metrics = jax.eval_shape(
        create_and_compute_loss,
        jax.random.key(9),
        observation,
        actions,
    )

    assert loss.shape == (1, 2)
    assert metrics["action_loss"].shape == ()
    assert metrics["caption_loss"].shape == ()
    assert metrics["caption_action_gate"].shape == ()
    assert metrics["caption_action_gate"].dtype == jnp.float32


def test_caption_input_spec():
    config = pi0_config.Pi0Config(
        use_support_context=True,
        use_caption_supervision=True,
        num_support_frames=2,
        caption_max_len=17,
    )

    observation, _ = config.inputs_spec(batch_size=3)

    assert observation.caption_input_tokens.shape == (3, 17)
    assert observation.caption_input_mask.shape == (3, 17)
    assert observation.caption_target_tokens.shape == (3, 17)
    assert observation.caption_loss_mask.shape == (3, 17)
    assert observation.caption_hand_side_mask.shape == (3, 17)


def test_caption_config_rejects_negative_loss_weight():
    with np.testing.assert_raises_regex(ValueError, "caption_loss_weight"):
        pi0_config.Pi0Config(caption_loss_weight=-0.1)


def test_caption_config_rejects_zero_queries():
    with np.testing.assert_raises_regex(ValueError, "num_caption_queries"):
        pi0_config.Pi0Config(num_caption_queries=0)


def test_caption_config_rejects_invalid_robot_token_grid():
    with np.testing.assert_raises_regex(ValueError, "caption_robot_tokens_per_image"):
        pi0_config.Pi0Config(caption_robot_tokens_per_image=63)
