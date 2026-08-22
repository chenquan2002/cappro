import jax
import numpy as np

from openpi.models import model
from openpi.shared import download
from openpi.training import weight_loaders


def test_checkpoint_loader_allows_new_lora_support_and_caption_parameters(monkeypatch):
    reference = {
        "base": jax.ShapeDtypeStruct((2,), np.float32),
        "lora_adapter": jax.ShapeDtypeStruct((1,), np.float32),
        "support_role_embeddings": jax.ShapeDtypeStruct((5, 8), np.float32),
        "caption_query_tokens": jax.ShapeDtypeStruct((4, 8), np.float32),
        "caption_action_gate": jax.ShapeDtypeStruct((), np.float32),
        "unrelated_new_parameter": jax.ShapeDtypeStruct((1,), np.float32),
    }
    loaded = {"base": np.asarray([1.0, 2.0], dtype=np.float32)}
    loader = weight_loaders.CheckpointWeightLoader("unused")
    monkeypatch.setattr(download, "maybe_download", lambda path: path)
    monkeypatch.setattr(model, "restore_params", lambda *args, **kwargs: loaded)

    merged = loader.load(reference)

    assert set(merged) == {
        "base",
        "lora_adapter",
        "support_role_embeddings",
        "caption_query_tokens",
        "caption_action_gate",
    }
    np.testing.assert_array_equal(merged["base"], loaded["base"])
    assert isinstance(merged["lora_adapter"], jax.ShapeDtypeStruct)
    assert isinstance(merged["support_role_embeddings"], jax.ShapeDtypeStruct)


def test_checkpoint_loader_keeps_new_role_embedding_shape(monkeypatch):
    reference = {
        "support_role_embeddings": np.zeros((5, 8), dtype=np.float32),
    }
    loaded = {
        "support_role_embeddings": np.ones((4, 8), dtype=np.float32),
    }
    loader = weight_loaders.CheckpointWeightLoader("unused")
    monkeypatch.setattr(download, "maybe_download", lambda path: path)
    monkeypatch.setattr(model, "restore_params", lambda *args, **kwargs: loaded)

    merged = loader.load(reference)

    np.testing.assert_array_equal(merged["support_role_embeddings"], reference["support_role_embeddings"])
