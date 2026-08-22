import json

import numpy as np
import pytest

from openpi.policies import support_video


def _write_support(tmp_path, *, frames, progress):
    view_dir = tmp_path / "human" / "test_task" / "demo_clean" / "human_demo_000" / "front"
    view_dir.mkdir(parents=True)
    np.save(view_dir / "frames.npy", frames, allow_pickle=False)
    (view_dir / "meta.json").write_text(json.dumps({"progress": progress}), encoding="utf-8")
    return view_dir


def test_support_video_bank_loads_video_without_caption(tmp_path):
    frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    _write_support(tmp_path, frames=frames, progress=[0.0, 1.0])
    bank = support_video.SupportVideoBank(tmp_path, num_frames=2)

    assert bank.discover("test_task", "demo_clean") == (("human_demo_000", "front"),)
    context = bank.load("test_task", "demo_clean", "human_demo_000", "front")

    assert set(context) == {"support_images", "support_image_mask", "support_frame_progress"}
    np.testing.assert_array_equal(context["support_images"], frames)
    np.testing.assert_array_equal(context["support_image_mask"], np.ones((2,), dtype=bool))
    np.testing.assert_allclose(context["support_frame_progress"], np.asarray([0.0, 1.0]))


def test_make_null_video_support_masks_all_frames():
    context = support_video.make_null_video_support(num_frames=2)

    assert set(context) == {"support_images", "support_image_mask", "support_frame_progress"}
    assert context["support_images"].shape == (2, 224, 224, 3)
    assert context["support_images"].dtype == np.uint8
    assert not context["support_image_mask"].any()
    np.testing.assert_array_equal(context["support_frame_progress"], np.zeros((2,), dtype=np.float32))


def test_support_video_bank_rejects_wrong_frame_count(tmp_path):
    frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    _write_support(tmp_path, frames=frames, progress=[0.0])
    bank = support_video.SupportVideoBank(tmp_path, num_frames=2)

    with pytest.raises(ValueError, match="Expected support frames shape"):
        bank.load("test_task", "demo_clean", "human_demo_000", "front")


def test_attach_video_support_rejects_caption_fields():
    context = {
        "support_images": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "support_image_mask": np.ones((2,), dtype=bool),
        "support_frame_progress": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    observation = support_video.attach_video_support({"state": np.zeros((14,))}, context, chunk_progress=0.4)

    np.testing.assert_allclose(observation["chunk_progress"], np.asarray([0.4]))
    assert not any("caption" in key for key in observation)

    with pytest.raises(ValueError, match="unexpected=.*caption_input_tokens"):
        support_video.attach_video_support(
            {},
            {**context, "caption_input_tokens": np.zeros((4,), dtype=np.int32)},
            chunk_progress=0.0,
        )
