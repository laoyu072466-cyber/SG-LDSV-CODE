#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sg_ldsv.cache import HiddenCache
from sg_ldsv.model import SGLDSVHead
from sg_ldsv.train import score_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache = HiddenCache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = SGLDSVHead(cache.hidden_size)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    device = torch.device(args.device)
    head = head.to(device=device, dtype=torch.float32)

    scores = score_cache(head, cache, device=device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for meta, score in zip(cache.metadata, scores):
            row = {
                "problem_id": meta["problem_id"],
                "candidate_index": int(meta["candidate_index"]),
                "label": int(meta["label"]),
                "score": float(score),
            }
            if "response_tokens" in meta:
                row["response_tokens"] = int(meta["response_tokens"])
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()