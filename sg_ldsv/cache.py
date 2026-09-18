[Reading 92 lines from start (total: 92 lines, 0 remaining)]

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PairRecord:
    positive_row: int
    negative_row: int
    problem_key: str


def problem_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class HiddenCache:
    """Read-only response-hidden-state cache."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest_path = self.directory / "cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "PASS":
            raise RuntimeError("cache manifest status is not PASS")
        if self.manifest.get("dtype") != "float16":
            raise RuntimeError("cache dtype must be float16")

        self.metadata = [
            json.loads(line)
            for line in (self.directory / "candidate_meta.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.offsets = np.load(self.directory / "offsets.npy", allow_pickle=False, mmap_mode="r")

        candidate_count = int(self.manifest["candidate_count"])
        total_tokens = int(self.manifest["total_response_tokens"])
        shape = self.manifest.get("shape")
        if "hidden_size" in self.manifest:
            hidden_size = int(self.manifest["hidden_size"])
        elif isinstance(shape, list) and len(shape) == 2:
            hidden_size = int(shape[1])
        else:
            raise RuntimeError("manifest must contain hidden_size or shape=[tokens,hidden]")

        if len(self.metadata) != candidate_count:
            raise RuntimeError("candidate metadata count mismatch")
        if self.offsets.shape != (candidate_count + 1,):
            raise RuntimeError("offset count mismatch")
        if int(self.offsets[-1]) != total_tokens:
            raise RuntimeError("offset total does not match manifest")

        self.hidden_size = hidden_size
        self.hidden = np.memmap(
            self.directory / "hidden.f16.mmap",
            dtype=np.dtype("<f2"),
            mode="r",
            shape=(total_tokens, hidden_size),
        )
        self.lengths = np.diff(self.offsets).astype(np.int64)
        if np.any(self.lengths <= 0):
            raise RuntimeError("cache contains an empty response sequence")

    def __len__(self) -> int:
        return len(self.metadata)

    def sequence(self, row: int) -> np.ndarray:
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return np.asarray(self.hidden[start:end])


def collate_candidates(cache: HiddenCache, rows: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate empty row list")
    lengths = [int(cache.lengths[row]) for row in rows]
    maximum = max(lengths)
    hidden = torch.zeros((len(rows), maximum, cache.hidden_size), dtype=torch.float32)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for batch_row, (row, length) in enumerate(zip(rows, lengths)):
        value = np.array(cache.sequence(row), dtype=np.float32, copy=True)
        hidden[batch_row, :length] = torch.from_numpy(value)
        mask[batch_row, :length] = True
    return hidden, mask

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]