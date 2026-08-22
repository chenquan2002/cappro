import json
from pathlib import Path

import numpy as np
import pytest

from openpi.training import support_context


class _TinyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"value": np.asarray(index, dtype=np.int32)}


def _write_manifest(tmp_path, *, null_support: bool = False):
    frames_path = tmp_path / "frames.npy"
    np.save(frames_path, np.zeros((2, 224, 224, 3), dtype=np.uint8))
    record = {
        "global_episode_index": 0,
        "support_round_id": 0,
        "episode_length": 5,
        "support_type": "null" if null_support else "human",
        "has_support": not null_support,
        "support_frames_npy": "" if null_support else str(frames_path),
        "support_frame_progress": [0.0, 1.0],
        "video_caption": "The left hand presses the button.",
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest_path


def _write_phase_manifest(tmp_path, **phase_overrides):
    manifest_path = _write_manifest(tmp_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["episode_length"] = 101
    record.update(
        {
            "caption_phase_schema": "task_progress_v1",
            "caption_phase_boundaries": [0.58, 0.79],
            "video_phase_captions": ["phase one", "phase two", "phase three"],
            **phase_overrides,
        }
    )
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest_path


def _write_multiview_manifest(tmp_path):
    demo_dir = tmp_path / "human_demo_000"
    progress_by_view = {
        "front": [0.0, 1.0],
        "ego": [0.0, 0.75],
    }
    for pixel_value, (view, progress) in enumerate(progress_by_view.items(), start=1):
        view_dir = demo_dir / view
        view_dir.mkdir(parents=True)
        frames_path = view_dir / "frames.npy"
        np.save(frames_path, np.full((2, 224, 224, 3), pixel_value, dtype=np.uint8))
        (view_dir / "meta.json").write_text(
            json.dumps({"view": view, "progress": progress, "frames_npy": str(frames_path)}),
            encoding="utf-8",
        )

    record = {
        "global_episode_index": 0,
        "support_round_id": 0,
        "episode_length": 5,
        "support_type": "human",
        "has_support": True,
        "support_view": "front",
        "support_frames_npy": str(demo_dir / "front" / "frames.npy"),
        "support_frame_progress": progress_by_view["front"],
        "video_caption": "The left hand presses the button.",
    }
    manifest_path = tmp_path / "multiview_manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest_path


def test_support_round_dataset():
    dataset = support_context.SupportRoundDataset(_TinyDataset(), support_rounds_per_cycle=3)

    assert len(dataset) == 6
    assert dataset[4]["value"] == 0
    assert dataset[4]["support_round_id"] == 2


def test_add_support_context_builds_shifted_caption(tmp_path):
    transform = support_context.AddSupportContext(
        _write_manifest(tmp_path),
        num_support_frames=2,
        caption_max_len=24,
    )
    output = transform(
        {
            "episode_index": np.asarray(0),
            "frame_index": np.asarray(2),
            "support_round_id": np.asarray(0),
        }
    )

    token_count = int(output["caption_loss_mask"].sum())
    assert output["support_images"].shape == (2, 224, 224, 3)
    assert output["support_image_mask"].all()
    assert output["chunk_progress"] == np.asarray([0.5], dtype=np.float32)
    assert np.array_equal(
        output["caption_input_tokens"][1:token_count],
        output["caption_target_tokens"][: token_count - 1],
    )
    assert output["caption_hand_side_mask"].sum() == 1


@pytest.mark.parametrize(
    ("frame_index", "expected_caption"),
    [
        (0, "phase one"),
        (57, "phase one"),
        (58, "phase two"),
        (78, "phase two"),
        (79, "phase three"),
        (100, "phase three"),
    ],
)
def test_phase_caption_is_selected_from_training_progress(tmp_path, frame_index, expected_caption):
    transform = support_context.AddSupportContext(
        _write_phase_manifest(tmp_path),
        num_support_frames=2,
        caption_max_len=24,
    )
    transform._add_caption = lambda data, caption: data.update(selected_caption=caption)  # noqa: SLF001

    output = transform({"episode_index": 0, "frame_index": frame_index})

    assert output["selected_caption"] == expected_caption


def test_legacy_manifest_falls_back_to_full_video_caption(tmp_path):
    transform = support_context.AddSupportContext(
        _write_manifest(tmp_path),
        num_support_frames=2,
        caption_max_len=24,
    )
    transform._add_caption = lambda data, caption: data.update(selected_caption=caption)  # noqa: SLF001

    output = transform({"episode_index": 0, "frame_index": 2})

    assert output["selected_caption"] == "The left hand presses the button."


@pytest.mark.parametrize(
    "phase_overrides",
    [
        {"caption_phase_schema": "unknown"},
        {"caption_phase_boundaries": [0.79, 0.58]},
        {"video_phase_captions": ["phase one", "phase two"]},
        {"video_phase_captions": ["phase one", "", "phase three"]},
    ],
)
def test_manifest_rejects_invalid_phase_caption_metadata(tmp_path, phase_overrides):
    with pytest.raises(ValueError, match="Invalid phase caption metadata"):
        support_context.SupportManifest(_write_phase_manifest(tmp_path, **phase_overrides))


def test_null_support_masks_video_and_caption(tmp_path):
    transform = support_context.AddSupportContext(
        _write_manifest(tmp_path, null_support=True),
        num_support_frames=2,
        caption_max_len=24,
    )
    output = transform({"episode_index": 0, "frame_index": 0})

    assert not output["support_image_mask"].any()
    assert not output["caption_input_mask"].any()
    assert not output["caption_loss_mask"].any()


def test_support_view_override_remaps_frames_and_progress(tmp_path):
    manifest_path = _write_multiview_manifest(tmp_path)
    transform = support_context.AddSupportContext(
        manifest_path,
        num_support_frames=2,
        caption_max_len=24,
        support_view_override="ego",
    )

    output = transform({"episode_index": 0, "frame_index": 0})
    mapped_record = transform.manifest.get(0, 0)

    assert np.all(output["support_images"] == 2)
    np.testing.assert_array_equal(
        output["support_frame_progress"],
        np.asarray([0.0, 0.75], dtype=np.float32),
    )
    assert mapped_record["support_original_view"] == "front"
    assert mapped_record["support_view"] == "ego"
    assert Path(mapped_record["support_frames_npy"]).parent.name == "ego"


def test_support_view_override_none_preserves_manifest_selection(tmp_path):
    manifest_path = _write_multiview_manifest(tmp_path)
    transform = support_context.AddSupportContext(
        manifest_path,
        num_support_frames=2,
        caption_max_len=24,
        support_view_override="none",
    )

    output = transform({"episode_index": 0, "frame_index": 0})

    assert np.all(output["support_images"] == 1)
    np.testing.assert_array_equal(
        output["support_frame_progress"],
        np.asarray([0.0, 1.0], dtype=np.float32),
    )


def test_support_view_override_rejects_invalid_value(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "support_view_override"):
        support_context.SupportManifest(_write_manifest(tmp_path), support_view_override="overhead")


def test_support_view_override_missing_target_becomes_null(tmp_path):
    manifest_path = _write_multiview_manifest(tmp_path)
    (tmp_path / "human_demo_000" / "ego" / "frames.npy").unlink()

    transform = support_context.AddSupportContext(
        manifest_path,
        num_support_frames=2,
        caption_max_len=24,
        support_view_override="ego",
    )
    output = transform({"episode_index": 0, "frame_index": 0})
    mapped_record = transform.manifest.get(0, 0)

    assert not output["support_image_mask"].any()
    assert not output["caption_loss_mask"].any()
    assert mapped_record["support_type"] == "null"
    assert not mapped_record["has_support"]
    assert "support_skip_reason" in mapped_record


def test_support_file_removed_after_manifest_load_becomes_null(tmp_path):
    manifest_path = _write_multiview_manifest(tmp_path)
    transform = support_context.AddSupportContext(
        manifest_path,
        num_support_frames=2,
        caption_max_len=24,
        support_view_override="none",
    )
    (tmp_path / "human_demo_000" / "front" / "frames.npy").unlink()

    output = transform({"episode_index": 0, "frame_index": 0})

    assert not output["support_image_mask"].any()
    assert not output["caption_loss_mask"].any()
