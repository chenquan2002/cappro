import argparse
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _load_observation(policy, image_path: pathlib.Path, prompt: str) -> _model.Observation:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    image_chw = np.transpose(image, (2, 0, 1))
    raw_observation = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": image_chw,
            "cam_left_wrist": image_chw,
            "cam_right_wrist": image_chw,
        },
        "prompt": prompt,
    }

    inputs = policy._input_transform(raw_observation)  # noqa: SLF001
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


def _read_last_prefix_token(model: _pi0.Pi0, observation: _model.Observation, top_k: int):
    graphdef, state = nnx.split(model)

    @jax.jit
    def run(model_state, obs):
        restored_model = nnx.merge(graphdef, model_state)
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = restored_model.embed_prefix(obs)
        attention_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = restored_model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attention_mask,
            positions=positions,
        )

        last_valid_index = jnp.maximum(jnp.sum(prefix_mask, axis=1, dtype=jnp.int32) - 1, 0)
        hidden = prefix_out[jnp.arange(prefix_out.shape[0]), last_valid_index][:, None, :]
        logits = restored_model.PaliGemma.llm(hidden, method="decode")[:, 0]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        top_log_probs, top_token_ids = jax.lax.top_k(log_probs, top_k)
        return top_token_ids, top_log_probs, last_valid_index, jnp.linalg.norm(hidden, axis=-1)

    return run(state, observation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a real Pi0.5 prefix_out through the tied LM head.")
    parser.add_argument("--config", default="pi05_aloha_robotwin_lora")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--prompt", default="click the alarm clock")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        robotwin_repo_id=args.asset_id,
    )
    if not isinstance(policy._model, _pi0.Pi0):  # noqa: SLF001
        raise TypeError("This diagnostic requires the JAX Pi0/Pi0.5 model.")

    observation = _load_observation(policy, args.image, args.prompt)
    token_ids, log_probs, position, hidden_norm = _read_last_prefix_token(
        policy._model,  # noqa: SLF001
        observation,
        args.top_k,
    )
    token_ids, log_probs, position, hidden_norm = jax.device_get((token_ids, log_probs, position, hidden_norm))

    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
    sentencepiece = tokenizer._tokenizer  # noqa: SLF001

    print(f"prompt: {args.prompt}")
    print(f"image: {args.image}")
    print(f"last valid prefix position: {int(position[0])}")
    print(f"prefix hidden norm: {float(hidden_norm[0, 0]):.6f}")
    print("top tied-head tokens:")
    for rank, (raw_token_id, log_prob) in enumerate(zip(token_ids[0], log_probs[0], strict=True), start=1):
        token_id = int(raw_token_id)
        piece = sentencepiece.id_to_piece(token_id)
        text = sentencepiece.decode([token_id])
        probability = float(np.exp(log_prob))
        print(f"{rank:2d}. id={token_id:6d} prob={probability:.8f} piece={piece!r} text={text!r}")


if __name__ == "__main__":
    main()
