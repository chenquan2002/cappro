import numpy as np

from openpi.policies import aloha_policy


def test_aloha_inference_preserves_video_support_without_caption():
    support_images = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    data = {
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "state": np.zeros((14,), dtype=np.float32),
        "prompt": "press the alarm clock",
        "support_images": support_images,
        "support_image_mask": np.ones((2,), dtype=bool),
        "support_frame_progress": np.asarray([0.0, 1.0], dtype=np.float32),
        "chunk_progress": np.asarray([0.25], dtype=np.float32),
    }

    transformed = aloha_policy.AlohaInputs(adapt_to_pi=False)(data)

    np.testing.assert_array_equal(transformed["support_images"], support_images)
    np.testing.assert_array_equal(transformed["support_image_mask"], np.ones((2,), dtype=bool))
    np.testing.assert_allclose(transformed["support_frame_progress"], np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(transformed["chunk_progress"], np.asarray([0.25]))
    assert not any(key.startswith("caption_") for key in transformed)
