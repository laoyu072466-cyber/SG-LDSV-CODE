#!/usr/bin/env python
from __future__ import annotations

import argparse

from sg_ldsv.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 456])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        train_cache_dir=args.train_cache,
        val_cache_dir=args.val_cache,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()