import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_caption_teacher_forcing():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=24)
    inputs, input_mask, targets, loss_mask, hand_side_mask = tokenizer.tokenize_caption_teacher_forcing(
        "The left hand presses the button."
    )

    token_count = int(loss_mask.sum())
    assert inputs.shape == (24,)
    assert targets.shape == (24,)
    assert input_mask.dtype == np.bool_
    assert np.array_equal(input_mask, loss_mask)
    assert np.array_equal(inputs[1:token_count], targets[: token_count - 1])
    assert hand_side_mask.sum() == 1
    assert not hand_side_mask[token_count:].any()


def test_caption_hand_side_mask_ignores_spatial_relation():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=24)
    *_, hand_side_mask = tokenizer.tokenize_caption_teacher_forcing("Place the object on the left side of the tray.")

    assert not hand_side_mask.any()


def test_caption_teacher_forcing_rejects_truncation():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=2)

    with pytest.raises(ValueError, match="exceeds max length"):
        tokenizer.tokenize_caption_teacher_forcing("This caption is longer than two tokens.")


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
