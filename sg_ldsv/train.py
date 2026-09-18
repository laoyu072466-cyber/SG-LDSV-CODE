from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cache import HiddenCache, collate_candidates
from .metrics import selection_pair_metrics
from .model import SGLDSVHead, trainable_parameter_count
from .pairs import build_problem_pairs


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PAIR_BATCH_SIZE = 32
MAX_EPOCHS = 5
PATIENCE = 2
GRADIENT_CLIP = 1.0


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def score_cache(
    head: SGLDSVHead,
    cache: HiddenCache,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    scores = np.empty(len(cache), dtype=np.float64)
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            rows = list(range(start, min(start + batch_size, len(cache))))
            hidden, mask = collate_candidates(cache, rows)
            output = head(
                hidden.to(device=device, dtype=torch.float32, non_blocking=True),
                mask.to(device=device, non_blocking=True),
            )
            value = output["score"].detach().cpu().double().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("non-finite verifier score")
            scores[start:start + len(rows)] = value
    return scores


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def train(
    *,
    train_cache_dir: str | Path,
    val_cache_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str = "cuda",
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(output_dir)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "history.json"
    result_path = output_dir / "run_result.json"

    seed_everything(seed)
    train_cache = HiddenCache(train_cache_dir)
    val_cache = HiddenCache(val_cache_dir)
    if train_cache.hidden_size != val_cache.hidden_size:
        raise RuntimeError("train/val hidden-size mismatch")

    train_pairs, train_pair_stats = build_problem_pairs(train_cache.metadata, seed=seed)
    val_pairs, val_pair_stats = build_problem_pairs(val_cache.metadata, seed=seed)

    torch_device = torch.device(device)
    head = SGLDSVHead(train_cache.hidden_size).to(torch_device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro = -math.inf
    best_micro = -math.inf
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.monotonic()
        head.train()
        shuffled = list(train_pairs)
        random.Random(seed + epoch * 1_000_003).shuffle(shuffled)

        loss_sum = 0.0
        pair_count = 0
        for batch_pairs in _chunks(shuffled, PAIR_BATCH_SIZE):
            rows = [x.positive_row for x in batch_pairs] + [x.negative_row for x in batch_pairs]
            hidden, mask = collate_candidates(train_cache, rows)
            hidden = hidden.to(torch_device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(torch_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            scores = head(hidden, mask)["score"]
            count = len(batch_pairs)
            loss = F.softplus(-(scores[:count] - scores[count:])).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()

            loss_sum += float(loss.detach().cpu()) * count
            pair_count += count

        val_scores = score_cache(head, val_cache, device=torch_device)
        val_macro, val_micro = selection_pair_metrics(val_scores, val_pairs)
        record = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / pair_count,
            "val_pair_macro": val_macro,
            "val_pair_micro": val_micro,
            "lambda_local": float(head.lambda_local.detach().cpu()),
            "epoch_wall_seconds": time.monotonic() - started,
        }
        history.append(record)

        improved = val_macro > best_macro
        if improved:
            best_epoch = epoch
            best_macro = val_macro
            best_micro = val_micro
            stale_epochs = 0
        else:
            stale_epochs += 1

        payload = {
            "schema_version": "SG_LDSV_CLEAN_CHECKPOINT_V1",
            "seed": seed,
            "epoch": epoch,
            "hidden_size": train_cache.hidden_size,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_pair_macro": best_macro,
            "best_val_pair_micro": best_micro,
            "stale_epochs": stale_epochs,
        }
        _atomic_torch(last_path, payload)
        if improved:
            _atomic_torch(best_path, payload)
        _write_json(log_path, history)

        print(
            f"epoch={epoch} loss={record['train_loss']:.8f} "
            f"val_pair_macro={val_macro:.8f} val_pair_micro={val_micro:.8f}",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break

    result = {
        "status": "PASS",
        "method": "sg_ldsv",
        "seed": seed,
        "hidden_size": train_cache.hidden_size,
        "trainable_parameters": trainable_parameter_count(head),
        "best_epoch": best_epoch,
        "best_val_pair_macro": best_macro,
        "best_val_pair_micro": best_micro,
        "epochs_completed": len(history),
        "train_pairs": train_pair_stats,
        "validation_pairs": val_pair_stats,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    _write_json(result_path, result)
    return result