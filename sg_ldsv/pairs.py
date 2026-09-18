from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Sequence

from .cache import PairRecord, problem_key


PAIR_CAP_PER_PROBLEM = 256


def build_problem_pairs(
    metadata: Sequence[dict[str, Any]],
    *,
    seed: int,
    cap: int = PAIR_CAP_PER_PROBLEM,
) -> tuple[list[PairRecord], dict[str, int]]:
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for row, item in enumerate(metadata):
        label = int(item["label"])
        if label not in (0, 1):
            raise ValueError(f"metadata row {row}: label must be 0/1")
        grouped[problem_key(item["problem_id"])][label].append(row)

    pairs: list[PairRecord] = []
    mixed = all_correct = all_wrong = capped = uncapped_total = 0

    for key in sorted(grouped):
        positives = grouped[key][1]
        negatives = grouped[key][0]
        if not positives:
            all_wrong += 1
            continue
        if not negatives:
            all_correct += 1
            continue

        mixed += 1
        cartesian = [(p, n) for p in positives for n in negatives]
        uncapped_total += len(cartesian)
        if len(cartesian) > cap:
            digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
            local_seed = int.from_bytes(digest[:8], "big")
            chosen = sorted(random.Random(local_seed).sample(range(len(cartesian)), cap))
            cartesian = [cartesian[i] for i in chosen]
            capped += 1
        pairs.extend(PairRecord(p, n, key) for p, n in cartesian)

    if not pairs:
        raise RuntimeError("pair dataset is empty")

    return pairs, {
        "problem_count": len(grouped),
        "mixed_problem_count": mixed,
        "all_correct_problem_count": all_correct,
        "all_wrong_problem_count": all_wrong,
        "capped_problem_count": capped,
        "uncapped_pair_count": uncapped_total,
        "final_pair_count": len(pairs),
    }

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]