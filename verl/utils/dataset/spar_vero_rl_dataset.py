"""Dataset adapter for cvis-tmu/spar-vero-rl."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


class SparVeroRLDataset(RLHFDataset):
    """Adapt SPAR-Vero rows and relative image paths to verl's RLHF schema."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ) -> None:
        image_root = config.get("image_root")
        if not image_root:
            raise ValueError("data.image_root is required for SparVeroRLDataset")
        self.image_root = Path(str(image_root)).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"SPAR image root does not exist: {self.image_root}")

        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def maybe_filter_out_long_prompts(self, dataframe=None):
        dataframe = self._filter_unusable_samples(dataframe)
        return super().maybe_filter_out_long_prompts(dataframe)

    def _filter_unusable_samples(self, dataframe):
        """Drop rows that cannot be trained under the current SPAR image dump.

        The HF parquet includes ScanNet-hash scenes missing from a local SPAR-7M
        dump. Drop those here so filter_overlong_prompts can tokenize the rest
        instead of crashing on missing files.
        """
        before = len(dataframe)
        image_key = self.image_key
        dropped_missing = 0

        def keep(doc) -> bool:
            nonlocal dropped_missing
            for raw_path in doc.get(image_key) or []:
                try:
                    self._resolve_image_path(raw_path)
                except FileNotFoundError:
                    dropped_missing += 1
                    return False
            return True

        dataframe = dataframe.filter(keep, desc="filter SPAR samples with missing images")
        after = len(dataframe)
        if dropped_missing:
            print(
                f"filtered {dropped_missing} SPAR samples with missing images "
                f"({after}/{before} remain)"
            )
        if after == 0:
            raise FileNotFoundError(
                f"No SPAR samples remain after dropping missing or overlong images under {self.image_root}"
            )
        return dataframe

    def _resolve_image_path(self, raw_path: Any) -> Path:
        if isinstance(raw_path, dict):
            raw_path = raw_path.get("path") or raw_path.get("image")
        relative_path = Path(str(raw_path))
        if relative_path.is_absolute():
            raise ValueError(f"SPAR image path must be relative: {relative_path}")

        resolved_path = (self.image_root / relative_path).resolve()
        try:
            resolved_path.relative_to(self.image_root)
        except ValueError as exc:
            raise ValueError(f"SPAR image path escapes data.image_root: {relative_path}") from exc

        if not resolved_path.is_file():
            raise FileNotFoundError(f"SPAR image file does not exist: {resolved_path}")
        return resolved_path

    def _build_messages(self, example: dict) -> list[dict]:
        problem = example.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("SPAR row requires a non-empty string `problem`")

        raw_images = example.get(self.image_key) or []
        image_records = [{"path": str(self._resolve_image_path(path))} for path in raw_images]
        placeholders = "\n".join("<image>" for _ in image_records)
        prompt = f"{placeholders}\n{problem}" if placeholders else problem

        example[self.prompt_key] = [{"role": "user", "content": prompt}]
        example[self.image_key] = image_records

        reward_model = dict(example.get("reward_model") or {})
        reward_model["ground_truth"] = example.get("answer")
        example["reward_model"] = reward_model

        extra_info = dict(example.get("extra_info") or {})
        for key in ("answer", "reward_type", "type", "id"):
            extra_info[key] = example.get(key)
        example["extra_info"] = extra_info

        return super()._build_messages(example)
