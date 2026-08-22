import torch

from openpi.models import model


def test_torch_support_video_uses_same_channel_layout_as_robot_images():
    batch_size = 2
    num_support_frames = 3
    data = {
        "image": {name: torch.zeros((batch_size, 8, 8, 3), dtype=torch.uint8) for name in model.IMAGE_KEYS},
        "image_mask": {name: torch.ones((batch_size,), dtype=torch.bool) for name in model.IMAGE_KEYS},
        "state": torch.zeros((batch_size, 4), dtype=torch.float32),
        "support_images": torch.zeros((batch_size, num_support_frames, 8, 8, 3), dtype=torch.uint8),
        "support_image_mask": torch.ones((batch_size, num_support_frames), dtype=torch.bool),
        "support_frame_progress": torch.zeros((batch_size, num_support_frames), dtype=torch.float32),
        "chunk_progress": torch.zeros((batch_size, 1), dtype=torch.float32),
    }

    observation = model.Observation.from_dict(data)

    assert observation.images[model.IMAGE_KEYS[0]].shape == (batch_size, 3, 8, 8)
    assert observation.support_images.shape == (batch_size, num_support_frames, 3, 8, 8)
