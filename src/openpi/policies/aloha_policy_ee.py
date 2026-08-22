import dataclasses
from typing import ClassVar
import numpy as np
from scipy.spatial.transform import Rotation as R
from openpi import transforms
import einops
# === 核心：自定义数学转换逻辑 ===

def _compute_delta_ee(current_state: np.ndarray, target_action: np.ndarray) -> np.ndarray:
    """
    Input:
        current_state: (B, 16) or (16,) 
        target_action: (B, T, 16) or (T, 16)
    Output:
        delta_action: (B, T, 14) or (T, 14)
    """
    # === 1. 维度兼容处理 ===
    # 判断是否是单样本输入 (T, D)
    is_unbatched = target_action.ndim == 2
    
    if is_unbatched:
        # 增加 Batch 维度: (T, 16) -> (1, T, 16)
        target_action = target_action[None, ...]
        
        # State: (16,) -> (1, 1, 16)
        if current_state.ndim == 1:
            current_state = current_state[None, None, :]
        elif current_state.ndim == 2:
             # 防御性编程：如果是 (1, 16)
             current_state = current_state[:, None, :]
    else:
        # 已有 Batch 维度: (B, T, 16)
        # State: (B, 16) -> (B, 1, 16)
        if current_state.ndim == 2:
            current_state = current_state[:, None, :]

    # 此时 target_action 必定是 (B, T, 16)，解包安全
    B, T, _ = target_action.shape
    
    # === 2. 原始计算逻辑 (保持不变) ===
    L_POS_IDX, L_QUAT_IDX, L_GRIP_IDX = slice(0, 3), slice(3, 7), 7
    R_POS_IDX, R_QUAT_IDX, R_GRIP_IDX = slice(8, 11), slice(11, 15), 15
    
    # Left Arm
    l_pos_curr = current_state[..., L_POS_IDX]
    l_pos_next = target_action[..., L_POS_IDX]
    l_delta_pos = l_pos_next - l_pos_curr

    l_quat_curr = current_state[..., L_QUAT_IDX]
    l_quat_next = target_action[..., L_QUAT_IDX]
    l_quat_curr = np.broadcast_to(l_quat_curr, l_quat_next.shape)
    
    def to_scipy(q): return np.roll(q, -1, axis=-1) 
    
    r_curr = R.from_quat(to_scipy(l_quat_curr.reshape(-1, 4)))
    r_next = R.from_quat(to_scipy(l_quat_next.reshape(-1, 4)))
    r_rel = r_next * r_curr.inv()
    l_delta_euler = r_rel.as_euler('xyz', degrees=False).reshape(B, T, 3)

    l_grip = target_action[..., L_GRIP_IDX:L_GRIP_IDX+1]

    # Right Arm
    r_pos_curr = current_state[..., R_POS_IDX]
    r_pos_next = target_action[..., R_POS_IDX]
    r_delta_pos = r_pos_next - r_pos_curr

    r_quat_curr = np.broadcast_to(current_state[..., R_QUAT_IDX], target_action[..., R_QUAT_IDX].shape)
    r_quat_next = target_action[..., R_QUAT_IDX]
    
    rr_curr = R.from_quat(to_scipy(r_quat_curr.reshape(-1, 4)))
    rr_next = R.from_quat(to_scipy(r_quat_next.reshape(-1, 4)))
    rr_rel = rr_next * rr_curr.inv()
    r_delta_euler = rr_rel.as_euler('xyz', degrees=False).reshape(B, T, 3)

    r_grip = target_action[..., R_GRIP_IDX:R_GRIP_IDX+1]

    delta_action = np.concatenate([
        l_delta_pos, l_delta_euler, l_grip,
        r_delta_pos, r_delta_euler, r_grip
    ], axis=-1)
    
    # === 3. 维度还原 ===
    if is_unbatched:
        return delta_action[0] # (1, T, 14) -> (T, 14)
    
    return delta_action

def _compute_absolute_ee(current_state: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
    """
    Inverse operation.
    """
    # === 1. 维度兼容处理 ===
    is_unbatched = delta_action.ndim == 2
    
    if is_unbatched:
        delta_action = delta_action[None, ...] # (1, T, 14)
        if current_state.ndim == 1:
            current_state = current_state[None, None, :]
    else:
        if current_state.ndim == 2:
            current_state = current_state[:, None, :]

    B, T, _ = delta_action.shape
    
    # === 2. 原始计算逻辑 (保持不变) ===
    # Left Arm
    l_pos_curr = current_state[..., 0:3]
    l_d_pos = delta_action[..., 0:3]
    l_abs_pos = l_pos_curr + l_d_pos
    
    l_quat_curr = np.broadcast_to(current_state[..., 3:7], (B, T, 4))
    l_d_euler = delta_action[..., 3:6]
    
    def to_scipy(q): return np.roll(q, -1, axis=-1)
    def from_scipy(q): return np.roll(q, 1, axis=-1)

    r_curr = R.from_quat(to_scipy(l_quat_curr.reshape(-1, 4)))
    r_delta = R.from_euler('xyz', l_d_euler.reshape(-1, 3), degrees=False)
    r_next = r_delta * r_curr 
    l_abs_quat = from_scipy(r_next.as_quat()).reshape(B, T, 4)
    
    l_abs_grip = delta_action[..., 6:7]

    # Right Arm
    r_pos_curr = current_state[..., 8:11]
    r_d_pos = delta_action[..., 7:10]
    r_abs_pos = r_pos_curr + r_d_pos
    
    r_quat_curr = np.broadcast_to(current_state[..., 11:15], (B, T, 4))
    r_d_euler = delta_action[..., 10:13]
    
    rr_curr = R.from_quat(to_scipy(r_quat_curr.reshape(-1, 4)))
    rr_delta = R.from_euler('xyz', r_d_euler.reshape(-1, 3), degrees=False)
    rr_next = rr_delta * rr_curr
    r_abs_quat = from_scipy(rr_next.as_quat()).reshape(B, T, 4)

    r_abs_grip = delta_action[..., 13:14]
    
    absolute_action = np.concatenate([
        l_abs_pos, l_abs_quat, l_abs_grip,
        r_abs_pos, r_abs_quat, r_abs_grip
    ], axis=-1)
    
    # === 3. 维度还原 ===
    if is_unbatched:
        return absolute_action[0]
    
    return absolute_action

# === Transforms 类定义 ===

@dataclasses.dataclass(frozen=True)
class AlohaInputs_EE(transforms.DataTransformFn):
    """
    EE Input Transform.
    Standardizes images (Uint8, HWC) and organizes state for Pi0.
    """
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        
        # === 核心修复: 图像预处理函数 ===
        # 这个函数负责把各种乱七八糟格式的输入图片统一成 OpenPI 喜欢的 uint8 HWC 格式
        def process_image(img):
            img = np.asarray(img)
            
            # 1. 维度调整 (Handle Channel-First)
            # 输入可能是 (T, C, H, W) 或 (C, H, W)
            # 我们需要 (T, H, W, C) 或 (H, W, C) 以便 PIL 处理
            if img.ndim == 4 and img.shape[1] == 3: 
                # (T, C, H, W) -> (T, H, W, C)
                img = einops.rearrange(img, "t c h w -> t h w c")
            elif img.ndim == 3 and img.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                img = einops.rearrange(img, "c h w -> h w c")
                
            # 2. 类型转换 (Float -> Uint8)
            # PIL requires uint8 for standard RGB
            if np.issubdtype(img.dtype, np.floating):
                # 假设 float 是 0.0-1.0 (LeRobot default) -> scale to 0-255
                # 使用 1.1 作为阈值以防轻微的浮点溢出
                if img.max() <= 1.1: 
                    img = (img * 255).clip(0, 255)
                img = img.astype(np.uint8)
            elif img.dtype != np.uint8:
                # 处理 int64/int32 -> uint8
                img = img.astype(np.uint8)
                
            return img
        # ==============================

        # 映射关系: Pi0 策略需要的 Key -> LeRobot 数据集里的 Key
        # 请根据你的 config repack_transforms 确保这里的 Key 是一致的
        cam_map = {
            "base_0_rgb": "cam_high",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }

        # 准备输出容器
        images = {}
        image_masks = {}
        
        # 1. 先处理主相机 (Base Image) 以获取参考尺寸
        base_source = cam_map["base_0_rgb"]
        if base_source in in_images:
            processed_base = process_image(in_images[base_source])
            images["base_0_rgb"] = processed_base
            image_masks["base_0_rgb"] = np.True_
            # 记录 shape 用于后续 fallback 生成黑图
            ref_shape = processed_base.shape
        else:
            raise ValueError(f"Base image '{base_source}' not found in input data.")

        # 2. 处理其他相机
        for dest, source in cam_map.items():
            if dest == "base_0_rgb": continue # 已处理
            
            if source in in_images:
                images[dest] = process_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                # Fallback: 全黑图像 (维度和类型必须与 Base Image 一致)
                images[dest] = np.zeros(ref_shape, dtype=np.uint8)
                image_masks[dest] = np.False_

        # 3. 组装输入
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]), # 16 dim
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class EEDeltaActions(transforms.DataTransformFn):
    """
    Converts Absolute EE Actions (16 dim) to Delta EE Actions (14 dim).
    Input: [Pos(3), Quat(4), Grip(1)] * 2
    Output: [D_Pos(3), D_Euler(3), Grip(1)] * 2
    """
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        
        state = data["state"] # (B, 16)
        actions = data["actions"] # (B, T, 16)
        
        # Apply the Math
        data["actions"] = _compute_delta_ee(state, actions)
        return data

@dataclasses.dataclass(frozen=True)
class EEAbsoluteActions(transforms.DataTransformFn):
    """
    Converts Delta EE Actions (14 dim) back to Absolute EE Actions (16 dim).
    Used during inference Output.
    """
    def __call__(self, data: dict) -> dict:
        # Inference output usually has 'actions' which are predicted deltas
        state = data["state"] # (B, 16) from input
        actions_delta = data["actions"] # (B, T, 14) from model
        
        data["actions"] = _compute_absolute_ee(state, actions_delta)
        return data

@dataclasses.dataclass(frozen=True)
class AlohaOutputs_EE(transforms.DataTransformFn):
    """
    Final Output wrapper. 
    Usually Pi0 expects 'actions' in the dict.
    This class can be used to format the final output if needed, 
    but EEAbsoluteActions usually does the heavy lifting.
    """
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"]}