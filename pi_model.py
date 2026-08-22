import os
from pathlib import Path

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.policies import support_video as _support_video
from openpi.training import config as _config


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


class PI0:
    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, usr_args=None):
        usr_args = dict(usr_args or {})
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = int(pi0_step)

        policy_root = Path(__file__).resolve().parent
        checkpoint_root = Path(
            usr_args.get("checkpoint_root")
            or os.environ.get("OPENPI_CHECKPOINT_ROOT", policy_root / "checkpoints")
        ).expanduser()
        checkpoint_path = checkpoint_root / train_config_name / model_name / str(checkpoint_id)
        assets_dir = checkpoint_path / "assets"
        assets_id = usr_args.get("asset_id")
        if assets_id is None:
            if not assets_dir.is_dir():
                raise FileNotFoundError(f"Checkpoint assets directory not found: {assets_dir}")
            asset_candidates = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
            if not asset_candidates:
                raise FileNotFoundError(f"No normalization assets found under: {assets_dir}")
            assets_id = asset_candidates[0]

        print("\n" + "=" * 80)
        print("\033[96m[权重加载信息]\033[0m")
        print(f"  训练配置: \033[93m{self.train_config_name}\033[0m")
        print(f"  模型名称: \033[93m{self.model_name}\033[0m")
        print(f"  检查点ID: \033[93m{self.checkpoint_id}\033[0m")
        print(f"  完整路径: \033[92m{checkpoint_path}\033[0m")
        print(f"  Assets ID: \033[94m{assets_id}\033[0m")
        print("=" * 80 + "\n")

        config = _config.get_config(train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            robotwin_repo_id=assets_id,
        )
        print("loading model success!")

        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        self._eval_frame_index = 0

        default_use_support = bool(getattr(config.model, "use_support_context", False))
        self.use_support_context = _as_bool(
            usr_args.get("use_support_context"),
            default=default_use_support,
        )
        self.support_context = None
        if self.use_support_context:
            self.num_support_frames = int(
                usr_args.get("num_support_frames") or getattr(config.model, "num_support_frames", 8)
            )
            self.mask_support_video = _as_bool(usr_args.get("mask_support_video"), default=False)
            support_bank_root = usr_args.get("support_bank_root") or os.environ.get(
                "SUPPORT_BANK_ROOT",
                str(policy_root.parents[1] / "data" / "support_data" / "support_bank_full"),
            )
            self.support_task_name = usr_args.get("support_task_name") or usr_args.get("task_name")
            self.support_task_config = usr_args.get("support_task_config") or usr_args.get("task_config")
            if not self.support_task_name or not self.support_task_config:
                raise ValueError("Support inference requires task_name and task_config")

            self.support_id = usr_args.get("support_id") or "human_demo_000"
            self.support_view = usr_args.get("support_view") or "front"
            self.random_support = _as_bool(usr_args.get("random_support"), default=False)
            support_seed = usr_args.get("support_seed")
            if support_seed is None:
                support_seed = usr_args.get("seed") or 0
            support_seed = int(support_seed)
            self.support_rng = np.random.default_rng(support_seed)
            self.support_bank = None
            self.support_candidates = ()
            if self.mask_support_video:
                print("[Support Eval] support video ablation enabled; skipping support bank discovery")
                self._load_support_context()
            else:
                self.support_bank = _support_video.SupportVideoBank(
                    support_bank_root,
                    num_frames=self.num_support_frames,
                )
                self.support_candidates = self.support_bank.discover(
                    self.support_task_name,
                    self.support_task_config,
                )
                print(f"[Support Eval] found {len(self.support_candidates)} video candidates")
            if self.random_support and not self.mask_support_video:
                self.resample_support_context()
            elif not self.mask_support_video:
                self._load_support_context()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def _load_support_context(self):
        if self.mask_support_video:
            self.support_context = _support_video.make_null_video_support(
                num_frames=self.num_support_frames,
            )
            print(
                "[Support Eval] support video masked: "
                f"frames={self.support_context['support_images'].shape}, "
                "support_image_mask=all_false"
            )
            return
        self.support_context = self.support_bank.load(
            self.support_task_name,
            self.support_task_config,
            self.support_id,
            self.support_view,
        )
        print(
            "[Support Eval] loaded video-only support: "
            f"{self.support_task_name}/{self.support_task_config}/{self.support_id}/{self.support_view}, "
            f"frames={self.support_context['support_images'].shape}, caption=disabled"
        )

    def resample_support_context(self):
        if not self.use_support_context or not self.random_support:
            return
        if self.mask_support_video:
            self._load_support_context()
            return
        if self.support_bank is None:
            raise RuntimeError("Support bank is unavailable")
        index = int(self.support_rng.integers(0, len(self.support_candidates)))
        self.support_id, self.support_view = self.support_candidates[index]
        self._load_support_context()

    def update_observation_window(self, img_arr, state, chunk_progress=None):
        img_front, img_right, img_left = (np.transpose(image, (2, 0, 1)) for image in img_arr[:3])

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
        if self.use_support_context:
            if self.support_context is None:
                raise RuntimeError("Support context has not been loaded")
            if chunk_progress is None:
                chunk_progress = self._eval_frame_index / max(self.pi0_step - 1, 1)
            self.observation_window = _support_video.attach_video_support(
                self.observation_window,
                self.support_context,
                chunk_progress=chunk_progress,
            )
        self._eval_frame_index += 1

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self._eval_frame_index = 0
        if self.use_support_context and self.random_support:
            self.resample_support_context()
        print("successfully unset obs and language intruction")
