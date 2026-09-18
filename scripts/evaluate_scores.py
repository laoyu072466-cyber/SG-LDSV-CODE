#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sg_ldsv.metrics import candidate_auroc, frozen_bestn, pair_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--frozen-subsets")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.scores).read_text(encoding="utf-8").splitlines()
        if line
    ]
    metadata = [
        {
            "problem_id": row["problem_id"],
            "candidate_index": row["candidate_index"],
            "label": row["label"],
        }
        for row in rows
    ]
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)

    pair_macro, pair_micro = pair_metrics(metadata, scores)
    result = {
        "pair_macro": pair_macro,
        "pair_micro": pair_micro,
        "auroc_candidate": candidate_auroc(metadata, scores),
    }

    if args.frozen_subsets:
        if not args.dataset:
            raise SystemExit("--dataset is required with --frozen-subsets")
        frozen = json.loads(Path(args.frozen_subsets).read_text(encoding="utf-8"))
        dataset_record = frozen["datasets"][args.dataset]
        result.update(frozen_bestn(metadata, scores, dataset_record))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()