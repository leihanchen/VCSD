#!/usr/bin/env python3
"""Precompute SPAR-Vero RL parquet with the same filters training applies.

Drops rows whose SPAR-7M images are missing, then drops rows whose Qwen-VL
chat+image token length exceeds --max-prompt-length. Writes original-schema
parquet (relative image paths) and optionally uploads to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/spar-vero-rl"))
    parser.add_argument("--source-repo", default=os.environ.get("SOURCE_DATASET_ID", "cvis-tmu/spar-vero-rl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/spar-vero-rl-filtered"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "8192")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SPAR_FILTER_WORKERS", "16")))
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_DATASET_ID", "cvis-tmu/spar-vero-rl-filtered"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def ensure_source_split(source_dir: Path, split: str, source_repo: str) -> Path:
    output = source_dir / f"{split}.parquet"
    if output.exists():
        return output
    from datasets import load_dataset

    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {source_repo} split={split} -> {output}", flush=True)
    dataset = load_dataset(source_repo, split=split)
    tmp = output.with_suffix(output.suffix + ".incomplete")
    try:
        dataset.to_parquet(tmp)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def load_tokenizer_processor(model_path: str, trust_remote_code: bool):
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    if processor is None:
        raise RuntimeError(
            f"hf_processor({model_path!r}) returned None; multimodal length "
            "filtering would not match training."
        )
    return tokenizer, processor


def filter_split(
    *,
    src: Path,
    tokenizer,
    processor,
    image_root: Path,
    max_prompt_length: int,
    num_workers: int,
) -> tuple[list[str], dict]:
    from omegaconf import OmegaConf

    from verl.utils.dataset.spar_vero_rl_dataset import SparVeroRLDataset

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(num_workers, cpu_count))
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "max_prompt_length": max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": workers,
            "image_root": str(image_root),
            "return_raw_chat": True,
            "truncation": "error",
        }
    )
    print(
        f"Filtering {src} with SparVeroRLDataset "
        f"(max_prompt_length={max_prompt_length}, workers={workers})",
        flush=True,
    )
    dataset = SparVeroRLDataset(
        data_files=str(src),
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )
    kept_ids = [str(x) for x in dataset.dataframe["id"]]
    stats = {
        "source_parquet": str(src),
        "source_rows": None,
        "kept_rows": len(kept_ids),
        "kept_ids_unique": len(set(kept_ids)),
    }
    return kept_ids, stats


def write_filtered_parquet(src: Path, dst: Path, kept_ids: list[str]) -> dict:
    import pandas as pd

    raw = pd.read_parquet(src)
    id_as_str = raw["id"].astype(str)
    kept_set = set(kept_ids)
    filtered = raw.loc[id_as_str.isin(kept_set)].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".incomplete")
    try:
        filtered.to_parquet(tmp, index=False)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    dropped = int(len(raw) - len(filtered))
    print(
        f"Wrote {dst} rows={len(filtered)}/{len(raw)} dropped={dropped}",
        flush=True,
    )
    return {
        "source_rows": int(len(raw)),
        "kept_rows": int(len(filtered)),
        "dropped_rows": dropped,
        "output_parquet": str(dst),
        "output_bytes": dst.stat().st_size,
    }


def write_readme(path: Path, payload: dict) -> None:
    splits = payload["splits"]
    split_lines = "\n".join(
        f"- `{name}`: {info['kept_rows']} / {info['source_rows']} rows kept "
        f"({info['dropped_rows']} dropped)"
        for name, info in splits.items()
    )
    path.write_text(
        "\n".join(
            [
                "---",
                "task_categories:",
                "- reinforcement-learning",
                "---",
                "",
                "# spar-vero-rl-filtered",
                "",
                "Pre-filtered subset of "
                f"[{payload['source_repo']}](https://huggingface.co/datasets/{payload['source_repo']}).",
                "",
                "Training-time `SparVeroRLDataset` filters applied once:",
                "",
                "1. Drop samples whose SPAR-7M images are missing under the precompute image root.",
                "2. Drop samples whose "
                f"`{payload['model_path']}` chat+image token length exceeds "
                f"{payload['max_prompt_length']}.",
                "",
                "Image paths stay relative (`SPAR-7M/...`). Training still needs the SPAR-7M tree.",
                "",
                "## Splits",
                "",
                split_lines,
                "",
                f"- Model: `{payload['model_path']}`",
                f"- max_prompt_length: {payload['max_prompt_length']}",
                f"- Created (UTC): {payload['created_at']}",
                "",
            ]
        )
        + "\n"
    )


def upload_to_hub(output_dir: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required for --upload")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.parquet", "README.md", "filter_stats.json"],
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Uploaded {output_dir} -> {url}", flush=True)
    return url


def main() -> int:
    args = parse_args()
    image_root = args.image_root.resolve()
    if not (image_root / "SPAR-7M").is_dir():
        print(f"ERROR: SPAR-7M image tree not found: {image_root / 'SPAR-7M'}", file=sys.stderr)
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tokenizer, processor = load_tokenizer_processor(args.model_path, args.trust_remote_code)

    payload = {
        "source_repo": args.source_repo,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "image_root": str(image_root),
        "model_path": args.model_path,
        "max_prompt_length": args.max_prompt_length,
        "num_workers": args.num_workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "hf_repo": args.hf_repo,
        "hf_url": None,
    }

    for split in splits:
        src = ensure_source_split(args.source_dir, split, args.source_repo)
        kept_ids, filter_stats = filter_split(
            src=src,
            tokenizer=tokenizer,
            processor=processor,
            image_root=image_root,
            max_prompt_length=args.max_prompt_length,
            num_workers=args.num_workers,
        )
        dst = args.output_dir / f"{split}.parquet"
        write_stats = write_filtered_parquet(src, dst, kept_ids)
        payload["splits"][split] = {**filter_stats, **write_stats}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_dir / "README.md", payload)
    stats_path = args.output_dir / "filter_stats.json"
    stats_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {stats_path}", flush=True)

    if args.upload:
        payload["hf_url"] = upload_to_hub(args.output_dir, args.hf_repo, args.private)
        stats_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
