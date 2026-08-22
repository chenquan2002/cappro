import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml


# Keep CLI style aligned with the user's existing scripts.
def parse_args_and_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs="*")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Keep override parsing style consistent with the user's current eval/rollout scripts.
    def parse_override_pairs(pairs):
        override_dict = {}
        if not pairs:
            return override_dict
        if len(pairs) % 2 != 0:
            raise ValueError("--overrides should be key value key value ...")
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("-")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    config.update(parse_override_pairs(args.overrides))
    return config


def build_round_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    run_name = cfg.get("run_name")
    if not run_name:
        raise ValueError("run_name is required")
    round_id = int(cfg.get("round_id", 0))

    experiment_root = Path(cfg.get("experiment_root", "experiments")) / run_name / f"round_{round_id}"
    rollout_root = experiment_root / "rollout"
    processed_root = experiment_root / "processed"
    return rollout_root, processed_root


def discover_episode_indices(rollout_root: Path) -> list[int]:
    data_dir = rollout_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"rollout data dir not found: {data_dir}")

    episode_indices = []
    pattern = re.compile(r"episode(\d+)\.hdf5$")
    for name in os.listdir(data_dir):
        m = pattern.match(name)
        if m:
            episode_indices.append(int(m.group(1)))
    episode_indices.sort()
    return episode_indices


def load_raw_hdf5(dataset_path: Path):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset does not exist at: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        image_dict = {}
        for cam_name in root["/observation"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs: list[np.ndarray]):
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("cv2.imencode failed")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def decode_and_resize(image_bytes: bytes, size=(640, 480)) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv2.resize(image, size)


def load_instruction_for_episode(rollout_root: Path, episode_idx: int, fallback: str = "") -> list[str]:
    instructions_path = rollout_root / "instructions.json"
    if not instructions_path.exists():
        return [fallback] if fallback else [f"episode_{episode_idx}"]

    with open(instructions_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    instructions = db.get("instructions", [])
    if episode_idx < len(instructions) and instructions[episode_idx]:
        return [instructions[episode_idx]]

    episode_map = db.get("episode_instructions", {})
    ep_instruction = episode_map.get(f"episode_{episode_idx}")
    if ep_instruction:
        return [ep_instruction]

    return [fallback] if fallback else [f"episode_{episode_idx}"]


def load_rollout_info(rollout_root: Path, episode_idx: int) -> dict[str, Any]:
    sidecar_path = rollout_root / "rollout_info" / f"episode{episode_idx}.json"
    if not sidecar_path.exists():
        return {"meta": {}, "steps": []}
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_episode(
    rollout_root: Path,
    processed_root: Path,
    episode_idx: int,
    fallback_task: str,
) -> None:
    raw_hdf5 = rollout_root / "data" / f"episode{episode_idx}.hdf5"
    left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_raw_hdf5(raw_hdf5)

    instructions = load_instruction_for_episode(rollout_root, episode_idx, fallback=fallback_task)
    rollout_info = load_rollout_info(rollout_root, episode_idx)

    episode_dir = processed_root / f"episode_{episode_idx}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    with open(episode_dir / "instructions.json", "w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, ensure_ascii=False, indent=2)

    qpos = []
    actions = []
    cam_high = []
    cam_right_wrist = []
    cam_left_wrist = []
    left_arm_dim = []
    right_arm_dim = []

    total_frames = int(left_gripper_all.shape[0])
    for j in range(total_frames):
        left_gripper = left_gripper_all[j]
        left_arm = left_arm_all[j]
        right_gripper = right_gripper_all[j]
        right_arm = right_arm_all[j]

        state = np.array(
            left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper],
            dtype=np.float32,
        )

        # Keep the same alignment logic as the original process_data_pi0.py:
        # qpos uses frames [0, ..., T-2], action uses states [1, ..., T-1].
        if j != total_frames - 1:
            qpos.append(state)
            cam_high.append(decode_and_resize(image_dict["head_camera"][j]))
            cam_right_wrist.append(decode_and_resize(image_dict["right_camera"][j]))
            cam_left_wrist.append(decode_and_resize(image_dict["left_camera"][j]))

        if j != 0:
            actions.append(state)
            left_arm_dim.append(left_arm.shape[0])
            right_arm_dim.append(right_arm.shape[0])

    num_samples = len(actions)
    if len(qpos) != num_samples:
        raise ValueError(
            f"alignment mismatch in episode {episode_idx}: qpos={len(qpos)} actions={num_samples}"
        )

    hdf5_path = episode_dir / f"episode_{episode_idx}.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=np.array(actions, dtype=np.float32))
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.array(qpos, dtype=np.float32))
        obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
        obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
        image = obs.create_group("images")

        cam_high_enc, len_high = images_encoding(cam_high)
        cam_right_enc, len_right = images_encoding(cam_right_wrist)
        cam_left_enc, len_left = images_encoding(cam_left_wrist)

        image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
        image.create_dataset("cam_right_wrist", data=cam_right_enc, dtype=f"S{len_right}")
        image.create_dataset("cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}")

    # Preserve rollout-only sidecar for later generate/value stages.
    processed_rollout_info = dict(rollout_info)
    processed_rollout_info.setdefault("meta", {})
    processed_rollout_info["meta"].update(
        {
            "episode_idx": episode_idx,
            "num_raw_frames": total_frames,
            "num_processed_samples": num_samples,
        }
    )

    # The rollout recorder writes one step per real env action, and the process stage
    # produces one (qpos_t, action_t) pair per executed step, so lengths should match.
    steps = processed_rollout_info.get("steps", [])
    if steps and len(steps) != num_samples:
        raise ValueError(
            f"rollout_info step mismatch in episode {episode_idx}: "
            f"steps={len(steps)} processed_samples={num_samples}"
        )

    with open(episode_dir / "rollout_info.json", "w", encoding="utf-8") as f:
        json.dump(processed_rollout_info, f, ensure_ascii=False, indent=2)


def write_processed_summary(processed_root: Path, episodes: list[int], cfg: dict[str, Any]) -> None:
    summary = {
        "run_name": cfg.get("run_name"),
        "round_id": int(cfg.get("round_id", 0)),
        "task_name": cfg.get("task_name"),
        "task_config": cfg.get("task_config"),
        "policy_name": cfg.get("policy_name"),
        "ckpt_setting": cfg.get("ckpt_setting"),
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    with open(processed_root / "processed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main(cfg: dict[str, Any]) -> None:
    rollout_root, processed_root = build_round_paths(cfg)
    processed_root.mkdir(parents=True, exist_ok=True)

    episode_indices = discover_episode_indices(rollout_root)
    fallback_task = str(cfg.get("task_name", ""))

    print(f"[Process] read rollout data from: {rollout_root}")
    print(f"[Process] write processed data to: {processed_root}")
    print(f"[Process] discovered {len(episode_indices)} episodes")

    for ep_idx in episode_indices:
        save_processed_episode(
            rollout_root=rollout_root,
            processed_root=processed_root,
            episode_idx=ep_idx,
            fallback_task=fallback_task,
        )
        print(f"[Process] episode_{ep_idx} success")

    write_processed_summary(processed_root, episode_indices, cfg)
    print(f"[Process] all done: {processed_root}")


if __name__ == "__main__":
    cfg = parse_args_and_config()
    main(cfg)
