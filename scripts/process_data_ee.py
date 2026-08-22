import sys
import os
import h5py
import numpy as np
import json
import cv2
import argparse
from pathlib import Path

def load_hdf5_ee(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        sys.exit(1)

    with h5py.File(dataset_path, "r") as root:
        left_endpose = root["/endpose/left_endpose"][()]
        right_endpose = root["/endpose/right_endpose"][()]
        left_gripper = root["/endpose/left_gripper"][()]
        right_gripper = root["/endpose/right_gripper"][()]

        image_dict = dict()
        if "observation" in root:
            for cam_name in root["/observation"].keys():
                if "rgb" in root[f"/observation/{cam_name}"]:
                    image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
        
    return left_endpose, left_gripper, right_endpose, right_gripper, image_dict


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        if success:
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        else:
            print(f"Warning: Image encoding failed for frame {i}")
    return encode_data, max_len


def data_transform_ee(path, episode_num, save_path):
    begin = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        # 1. 指令处理
        try:
            instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
            with open(instruction_data_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict.get("seen", instruction_dict.get("instructions", ["Default instruction"]))
            save_instructions_json = {"instructions": instructions}
        except FileNotFoundError:
            save_instructions_json = {"instructions": ["Do the task."]}

        episode_save_dir = os.path.join(save_path, f"episode_{i}")
        os.makedirs(episode_save_dir, exist_ok=True)

        with open(os.path.join(episode_save_dir, "instructions.json"), "w") as f:
            json.dump(save_instructions_json, f, indent=2)

        # 2. 读取数据
        try:
            hdf5_source_path = os.path.join(path, "data", f"episode{i}.hdf5")
            left_pose_all, left_grip_all, right_pose_all, right_grip_all, image_dict = load_hdf5_ee(hdf5_source_path)
        except Exception as e:
            print(f"Error loading episode {i}: {e}")
            continue

        states = []
        actions = [] 
        
        # === 新增/恢复：维度记录 ===
        left_arm_dim = []
        right_arm_dim = []

        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []

        data_len = left_pose_all.shape[0]

        for j in range(0, data_len):
            l_p = left_pose_all[j]   # (7,)
            r_p = right_pose_all[j]  # (7,)
            l_g = left_grip_all[j]
            r_g = right_grip_all[j]

            l_g_val = l_g if isinstance(l_g, (int, float)) else l_g.item()
            r_g_val = r_g if isinstance(r_g, (int, float)) else r_g.item()

            # State = [L_Pose(7), L_Grip(1), R_Pose(7), R_Grip(1)]
            current_vec = np.concatenate([l_p, [l_g_val], r_p, [r_g_val]]).astype(np.float32)

            if j != data_len - 1:
                states.append(current_vec)

                # 处理图像
                try:
                    if "head_camera" in image_dict:
                        img_bytes = image_dict["head_camera"][j]
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        cam_high.append(cv2.resize(img, (640, 480)))
                    if "right_camera" in image_dict:
                        img_bytes = image_dict["right_camera"][j]
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        cam_right_wrist.append(cv2.resize(img, (640, 480)))
                    if "left_camera" in image_dict:
                        img_bytes = image_dict["left_camera"][j]
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        cam_left_wrist.append(cv2.resize(img, (640, 480)))
                except Exception as e:
                    print(f"Error processing images at frame {j}: {e}")

            if j != 0:
                actions.append(current_vec)
                # === 记录维度 ===
                # 在 EE 模式下，"arm" 的维度固定为 7 (x,y,z,qw,qx,qy,qz)
                left_arm_dim.append(7)
                right_arm_dim.append(7)

        # 3. 保存
        target_hdf5_path = os.path.join(episode_save_dir, f"episode_{i}.hdf5")
        
        with h5py.File(target_hdf5_path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            
            obs = f.create_group("observations")
            obs.create_dataset("state", data=np.array(states))
            
            # === 保存维度信息，保持兼容性 ===
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            
            image_grp = obs.create_group("images")
            if len(cam_high) > 0:
                enc, length = images_encoding(cam_high)
                image_grp.create_dataset("cam_high", data=enc, dtype=f"S{length}")
            if len(cam_right_wrist) > 0:
                enc, length = images_encoding(cam_right_wrist)
                image_grp.create_dataset("cam_right_wrist", data=enc, dtype=f"S{length}")
            if len(cam_left_wrist) > 0:
                enc, length = images_encoding(cam_left_wrist)
                image_grp.create_dataset("cam_left_wrist", data=enc, dtype=f"S{length}")

        begin += 1
        print(f"Process {i} success! State Shape: {np.array(states).shape}")

    return begin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episodes to EE format.")
    parser.add_argument("task_name", type=str, default="beat_block_hammer")
    parser.add_argument("setting", type=str, default="default")
    parser.add_argument("expert_data_num", type=int, default=50)
    args = parser.parse_args()

    policy_root = Path(__file__).resolve().parents[1]
    robotwin_root = policy_root.parents[1]
    dataset_root = Path(
        os.environ.get("ROBOTWIN_DATASET_ROOT", robotwin_root / "data_NAS" / "data" / "dataset")
    ).expanduser()
    load_dir = dataset_root / args.task_name / args.setting
    if not load_dir.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {load_dir}. "
            "Set ROBOTWIN_DATASET_ROOT to the local RoboTwin dataset root."
        )
    print(f'Read data from path: {load_dir}')

    processed_data_root = Path(
        os.environ.get("OPENPI_PROCESSED_EE_DATA_ROOT", policy_root / "processed_data_ee")
    ).expanduser()
    target_dir = processed_data_root / f"{args.task_name}-{args.setting}-{args.expert_data_num}"
    data_transform_ee(str(load_dir), args.expert_data_num, str(target_dir))
