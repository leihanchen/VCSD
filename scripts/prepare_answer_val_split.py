#!/usr/bin/env python3
"""Create a deterministic train/validation split for answer-letter validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/train.parquet", help="Source parquet file.")
    parser.add_argument("--train-output", default="data/train_answer.parquet", help="Output train parquet file.")
    parser.add_argument("--val-output", default="data/val_answer.parquet", help="Output validation parquet file.")
    parser.add_argument("--val-size", type=int, default=256, help="Number of validation rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for row sampling.")
    return parser.parse_args()


def _has_answer(row: pd.Series) -> bool:
    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth"):
        return True
    return isinstance(extra_info, dict) and bool(extra_info.get("answer"))


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    df = pd.read_parquet(source)
    if df.empty:
        raise ValueError(f"{source} contains no rows")
    if args.val_size <= 0:
        raise ValueError("--val-size must be positive")
    if args.val_size >= len(df):
        raise ValueError(f"--val-size must be smaller than dataset size {len(df)}")

    missing_answer = df[~df.apply(_has_answer, axis=1)]
    if not missing_answer.empty:
        raise ValueError(f"{len(missing_answer)} rows do not contain reward_model.ground_truth or extra_info.answer")

    val_df = df.sample(n=args.val_size, random_state=args.seed)
    train_df = df.drop(index=val_df.index)

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.reset_index(drop=True).to_parquet(train_output, index=False)
    val_df.reset_index(drop=True).to_parquet(val_output, index=False)

    print(f"input rows: {len(df)}")
    print(f"train rows: {len(train_df)} -> {train_output}")
    print(f"val rows: {len(val_df)} -> {val_output}")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
