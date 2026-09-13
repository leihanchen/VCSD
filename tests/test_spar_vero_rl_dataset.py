from pathlib import Path

import datasets
import pytest
from PIL import Image

from verl.utils.dataset.spar_vero_rl_dataset import SparVeroRLDataset
from verl.utils.dataset.vision_utils import process_image


class _ShortPromptProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {"input_ids": [[0] * 8]}


def _config(image_root: Path) -> dict:
    return {
        "image_root": str(image_root),
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "filter_overlong_prompts": False,
        "return_multi_modal_inputs": True,
    }


def _write_rows(parquet_path: Path, rows: list[dict]) -> None:
    datasets.Dataset.from_list(rows).to_parquet(parquet_path)


def _row(image_path: str, sample_id: str = "sample") -> dict:
    return {
        "images": [image_path],
        "problem": "Question?",
        "answer": "A",
        "type": "depth_prediction_oc",
        "id": sample_id,
        "data_source": "spar7m",
        "reward_type": "multiple_choice",
    }


def test_dataset_exposes_multiview_prompt_and_reward_metadata(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    relative_paths = [
        "SPAR-7M/spar/scannet/images/scene/image_color/1.jpg",
        "SPAR-7M/spar/scannet/images/scene/image_color/2.jpg",
    ]
    for relative_path in relative_paths:
        image_path = image_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()

    parquet_path = tmp_path / "train.parquet"
    datasets.Dataset.from_list(
        [
            {
                "images": relative_paths,
                "problem": "What changed between the two views?",
                "answer": "move_left:1",
                "type": "view_change_infer",
                "id": "sample-1",
                "data_source": "spar7m",
                "reward_type": "string_match",
            }
        ]
    ).to_parquet(parquet_path)

    dataset = SparVeroRLDataset(
        data_files=[str(parquet_path)],
        tokenizer=object(),
        processor=object(),
        config=_config(image_root),
    )

    sample = dataset[0]

    content = sample["raw_prompt"][0]["content"]
    assert [part["type"] for part in content] == ["image", "text", "image", "text"]
    assert [part.get("text") for part in content if part["type"] == "text"] == [
        "\n",
        "\nWhat changed between the two views?",
    ]
    assert [part["path"] for part in content if part["type"] == "image"] == [
        str((image_root / relative_path).resolve()) for relative_path in relative_paths
    ]
    assert sample["reward_model"] == {"ground_truth": "move_left:1"}
    assert sample["extra_info"] == {
        "answer": "move_left:1",
        "reward_type": "string_match",
        "type": "view_change_infer",
        "id": "sample-1",
    }


def test_dataset_rejects_a_missing_image_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SPAR image root does not exist"):
        SparVeroRLDataset(
            data_files=[str(tmp_path / "unused.parquet")],
            tokenizer=object(),
            processor=object(),
            config=_config(tmp_path / "missing"),
        )


def test_dataset_skips_rows_with_missing_images(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    valid_path = "SPAR-7M/spar/scannet/images/scene/image_color/1.jpg"
    (image_root / valid_path).parent.mkdir(parents=True, exist_ok=True)
    (image_root / valid_path).touch()

    parquet_path = tmp_path / "train.parquet"
    _write_rows(
        parquet_path,
        [
            _row("SPAR-7M/missing.jpg", "missing"),
            _row(valid_path, "valid"),
        ],
    )

    dataset = SparVeroRLDataset(
        data_files=[str(parquet_path)],
        tokenizer=object(),
        processor=object(),
        config=_config(image_root),
    )

    assert len(dataset) == 1
    assert dataset[0]["extra_info"]["id"] == "valid"


@pytest.mark.parametrize(
    ("image_path", "error"),
    [
        ("/absolute/image.jpg", "must be relative"),
        ("../outside.jpg", "escapes data.image_root"),
        ("SPAR-7M/missing.jpg", "No SPAR samples remain"),
    ],
)
def test_dataset_rejects_unsafe_or_missing_image_paths(tmp_path: Path, image_path: str, error: str) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    parquet_path = tmp_path / "train.parquet"
    _write_rows(parquet_path, [_row(image_path)])

    with pytest.raises((ValueError, FileNotFoundError), match=error):
        SparVeroRLDataset(
            data_files=[str(parquet_path)],
            tokenizer=object(),
            processor=object(),
            config=_config(image_root),
        )


def _tiny_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(0, 128, 255)).save(path, format="JPEG")


def test_process_image_accepts_path_strings(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    _tiny_jpeg(image_path)

    image = process_image(str(image_path))

    assert image.mode == "RGB"
    assert image.size[0] > 0


def test_overlong_filter_keeps_spar_path_string_images(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    relative_path = "SPAR-7M/spar/scannet/images/scene/image_color/1.jpg"
    _tiny_jpeg(image_root / relative_path)

    parquet_path = tmp_path / "train.parquet"
    _write_rows(parquet_path, [_row(relative_path, "valid")])

    tokenizer = object()
    processor = _ShortPromptProcessor()

    config = _config(image_root)
    config["filter_overlong_prompts"] = True
    config["max_prompt_length"] = 32
    config["filter_overlong_prompts_workers"] = 1

    dataset = SparVeroRLDataset(
        data_files=[str(parquet_path)],
        tokenizer=tokenizer,
        processor=processor,
        config=config,
    )

    assert len(dataset) == 1
    assert dataset[0]["extra_info"]["id"] == "valid"
