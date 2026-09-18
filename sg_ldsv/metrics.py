from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from .cache import PairRecord, problem_key


def selection_pair_metrics(scores: np.ndarray, pairs: Sequence[PairRecord]) -> tuple[float, float]:
    """Validation metric used for checkpoint selection.

    This intentionally matches the frozen training code: a score tie counts as 0
    for selection, and Pair-Macro first averages within problem.
    """
    by_problem: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        by_problem[pair.problem_key].append(
            float(scores[pair.positive_row] > scores[pair.negative_row])
        )
    macro = float(np.mean([np.mean(v) for v in by_problem.values()]))
    micro = float(np.mean([x for v in by_problem.values() for x in v]))
    return macro, micro


def grouped_rows(metadata: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    raw_id: dict[str, Any] = {}
    for row, item in enumerate(metadata):
        key = problem_key(item["problem_id"])
        grouped[key].append(row)
        raw_id[key] = item["problem_id"]

    result = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda r: int(metadata[r]["candidate_index"]))
        indices = [int(metadata[r]["candidate_index"]) for r in rows]
        if indices != list(range(len(rows))):
            raise RuntimeError(f"candidate_index must be contiguous for problem {raw_id[key]!r}")
        result.append({"problem_key": key, "problem_id": raw_id[key], "rows": rows})
    return result


def pair_metrics(metadata: Sequence[dict[str, Any]], scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    groups = grouped_rows(metadata)
    total_value = 0.0
    total_pairs = 0
    macro_values = []

    for group in groups:
        positives = [r for r in group["rows"] if labels[r] == 1]
        negatives = [r for r in group["rows"] if labels[r] == 0]
        if not positives or not negatives:
            continue
        value = 0.0
        count = 0
        for p in positives:
            for n in negatives:
                diff = scores[p] - scores[n]
                value += 1.0 if diff > 0 else 0.5 if diff == 0 else 0.0
                count += 1
        total_value += value
        total_pairs += count
        macro_values.append(value / count)

    if not macro_values:
        raise RuntimeError("no mixed-label problems")
    return float(np.mean(macro_values)), total_value / total_pairs


def candidate_auroc(metadata: Sequence[dict[str, Any]], scores: np.ndarray) -> float:
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("AUROC requires both labels")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _tie_expected(labels: np.ndarray, scores: np.ndarray, rows: list[int]) -> float:
    selected = scores[rows]
    maximum = selected.max()
    top = [row for row in rows if scores[row] == maximum]
    return float(labels[top].mean())


def frozen_bestn(
    metadata: Sequence[dict[str, Any]],
    scores: np.ndarray,
    frozen_dataset: dict[str, Any],
) -> dict[str, float]:
    labels = np.asarray([int(x["label"]) for x in metadata], dtype=np.int8)
    groups = grouped_rows(metadata)
    frozen = {problem_key(x["problem_id"]): x for x in frozen_dataset["problems"]}
    result: dict[str, float] = {}

    n_values = sorted({int(n) for row in frozen_dataset["problems"] for n in row["subsets"]})
    for n in n_values:
        total = 0.0
        count = 0
        for group in groups:
            record = frozen[group["problem_key"]]
            for subset in record["subsets"][str(n)]:
                rows = [group["rows"][index] for index in subset]
                total += _tie_expected(labels, scores, rows)
                count += 1
        result[f"best_at_{n}"] = total / count

    result["best_at_full"] = float(
        np.mean([_tie_expected(labels, scores, group["rows"]) for group in groups])
    )
    return result