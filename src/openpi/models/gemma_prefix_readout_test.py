import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as _gemma
from openpi.shared import nnx_utils


class _JointGemmaHarness(nnx.Module):
    def __init__(self):
        config = _gemma.get_config("dummy")
        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[config, config],
                embed_dtype="float32",
            )
        )
        self.llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    def forward(self, prefix, suffix, mask, positions):
        (_, suffix_out), _ = self.llm(
            [prefix, suffix],
            mask=mask,
            positions=positions,
        )
        return suffix_out

    def forward_with_prefix_readout(self, prefix, suffix, mask, positions):
        (prefix_out, suffix_out), _ = self.llm(
            [prefix, suffix],
            mask=mask,
            positions=positions,
        )
        logits = self.llm(prefix_out[:, :1], method="decode")
        return suffix_out, logits


def test_joint_gemma_prefix_out_supports_tied_readout():
    model = _JointGemmaHarness()
    prefix = jax.random.normal(jax.random.key(1), (1, 3, 64))
    suffix = jax.random.normal(jax.random.key(2), (1, 2, 64))
    mask = jnp.ones((1, 5, 5), dtype=jnp.bool_)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]

    baseline_suffix = nnx_utils.module_jit(model.forward)(prefix, suffix, mask, positions)
    readout_suffix, logits = nnx_utils.module_jit(model.forward_with_prefix_readout)(prefix, suffix, mask, positions)

    assert logits.shape == (1, 1, _gemma.PALIGEMMA_VOCAB_SIZE)
    assert bool(jnp.all(jnp.isfinite(logits)))
    np.testing.assert_allclose(readout_suffix, baseline_suffix, rtol=0.0, atol=0.0)
