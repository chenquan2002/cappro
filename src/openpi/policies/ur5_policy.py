import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """
    Creates a random input example for the UR5 policy (RobotWin format).
    Useful for testing transforms or dry-running the model.
    """
    return {
        "state": np.random.rand(7),  # 6 joints + 1 gripper (RobotWin merged state)
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red cube",
    }


def _parse_image(image) -> np.ndarray:
    """
    Parses an image to the format expected by the model (H, W, C) and uint8.
    LeRobot datasets sometimes store images as float32 (C, H, W) or (H, W, C).
    """
    image = np.asarray(image)
    
    # If image is float (0-1), convert to uint8 (0-255)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # If image is channel-first (3, H, W), convert to channel-last (H, W, 3)
    # This handles the case where LeRobot might have already permuted the image
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
        
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """
    Transforms RobotWin UR5 data into the format expected by the Pi0 model.
    Expected input keys (mapped from LeRobotRepack):
        - 'state': (7,) array containing [joints(6) + gripper(1)]
        - 'base_rgb': Main camera image
        - 'wrist_rgb': Wrist camera image
        - 'prompt': Language instruction
        - 'actions': (T, 7) array (Training only)
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 1. State: RobotWin data already merges joints and gripper into 'state'
        # Shape: (7,) -> [joint_0, ..., joint_5, gripper]
        state = np.asarray(data["state"])

        # 2. Images: Parse and ensure correct format (H, W, C)
        # Note: The keys "base_rgb" and "wrist_rgb" come from LeRobotUR5DataConfig's RepackTransform.

        # base_image = _parse_image(data["base_rgb"])
        # wrist_image = _parse_image(data["wrist_rgb"])
        # print('DATA.KEY',list(data.keys()))
        # print('data',data)
        # images = data["images"]
        # base_image = _parse_image(images["cam_high"])
        # # 如果 evaluation 时不传 wrist，这里可以加个 get，但根据你的 config 应该都有
        # wrist_image = _parse_image(images["cam_left_wrist"]) 
        # print('IMAGE.KEY',list(images.keys()))
        #IMAGE.KEY ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        # print("jusge1", images['cam_left_wrist'] is None)
        # print("jusge1", images['cam_right_wrist'] is None)
        # # 1. 解析图片 (从 images 字典里取)
        # ['actions', 'base_rgb', 'prompt', 'state', 'wrist_rgb']
        base_image = _parse_image(data["base_rgb"])
        # 如果 evaluation 时不传 wrist，这里可以加个 get，但根据你的 config 应该都有
        wrist_image = _parse_image(data["wrist_rgb"])
      

        # 3. Create inputs dict for Pi0
        inputs = {
            "state": state,
            "image": {
                # Pi0's third-person view slot
                "base_0_rgb": base_image,
                
                # Pi0's left wrist view slot (UR5 typically uses this slot even if single arm)
                "left_wrist_0_rgb": wrist_image,
                
                # Pi0's right wrist view slot (Unused for single-arm UR5)
                # Must be padded with zeros to keep the model structure valid
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                # Mark which images are valid real data
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                
                # Mask out the unused right wrist so the model ignores it
                # Note: Pi0-FAST uses a different masking strategy (True) compared to standard Pi0 (False)
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 4. Actions: Only present during training
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # 5. Prompt: Language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """
    Transforms model outputs back into the dataset specific format during inference.
    """

    def __call__(self, data: dict) -> dict:
        # The model output might be padded (e.g., to 14 dims).
        # We strictly slice it to return only the first 7 dimensions (6 joints + 1 gripper).
        # return {"actions": np.asarray(data["actions"][:, :7])}
        action = np.asarray(data["actions"])
        padding = np.zeros_like(action)

        if action.ndim == 1:
            full_action = np.concatenate([action, padding], axis=0)
        else:
            full_action = np.concatenate([action, padding], axis=-1)
            
        return {"actions": full_action}