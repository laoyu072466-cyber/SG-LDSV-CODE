[Reading 140 lines from start (total: 140 lines, 0 remaining)]

# SG-LDSV

Clean reference implementation of **SG-LDSV**, a lightweight verifier for mathematical solution verification from frozen transformer hidden states.

**Manuscript:** *Beyond Endpoint Scoring: Gated Local Transition Statistics for Verifying Mathematical Solutions*

## Method

Given response-token hidden states (H=[h_1,ldots,h_T]) from one frozen transformer layer, SG-LDSV combines endpoint evidence with gated statistics of adjacent-token representation transitions.

```text
hidden states H [T,d]
  ├─ endpoint: h_T -> MLP -> endpoint score
  └─ local:
       H -> Linear(d,128,bias=False) -> LayerNorm
         -> Δz_t = z_{t+1} - z_t
         -> phi: 128 -> 32 -> ReLU
         -> gate_t = sigmoid(Linear(z_t))
         -> gated mean/std over valid transitions
         -> rho: 64 -> 96 -> ReLU -> 1
final score = endpoint score + lambda_local * local score
```

The gate is a learned weighting mechanism over valid adjacent-token transitions. The implementation does **not** require finetuning the language-model backbone.

## Repository scope

This repository intentionally contains only the clean SG-LDSV implementation and reproducibility utilities. It excludes private paths, experiment caches, checkpoints, generated candidates, logs, and unrelated follow-up experiments.

Included:

- SG-LDSV model head
- memory-mapped hidden-state cache reader
- problem-wise positive/negative pair construction
- frozen training protocol
- candidate scoring
- Pair-Macro / Pair-Micro / candidate AUROC / frozen Best@N evaluation
- small unit tests

## Frozen training protocol

The paper experiments used:

- frozen language-model backbone
- response hidden states from block 21 in the main SG-LDSV experiments
- AdamW, learning rate `1e-3`
- weight decay `1e-4`
- pair batch size `32`
- maximum `5` epochs
- early-stop patience `2`
- gradient clipping `1.0`
- FP32 verifier head
- at most `256` positive-negative pairs per problem
- seeds `42, 123, 456`
- checkpoint selection by validation Pair-Macro, strict improvement, earliest tie

The original frozen protocol hash was:

```text
29a1131a746c5f969469021bf970a8dfd02d651bc4435d0e4b4da6d6082c673b
```

## Hidden-state cache format

Each split is stored in one directory:

```text
cache_dir/
  cache_manifest.json
  candidate_meta.jsonl
  offsets.npy
  hidden.f16.mmap
```

`candidate_meta.jsonl` must contain at least:

```json
{"problem_id": 0, "candidate_index": 0, "label": 1}
```

The manifest must contain `status: "PASS"`, `dtype: "float16"`, `candidate_count`, `total_response_tokens`, and either `hidden_size` or a two-element `shape`.

The memory map stores all response-token hidden states consecutively in little-endian float16. `offsets.npy` has length `candidate_count + 1`.

## Install

```bash
python -m pip install -e .
```

The tested research environment used Python 3.12, PyTorch 2.7, NumPy 2.5, Transformers 4.55, and Tokenizers 0.21.

## Train

```bash
python scripts/train_sg_ldsv.py \
  --train-cache /path/to/gsm8k_train_cache \
  --val-cache /path/to/gsm8k_val_cache \
  --output-dir runs/gsm8k/seed_42 \
  --seed 42
```

The best checkpoint is selected only by validation Pair-Macro.

## Score a split

```bash
python scripts/score_sg_ldsv.py \
  --cache /path/to/gsm8k_test_cache \
  --checkpoint runs/gsm8k/seed_42/best.pt \
  --output scores/gsm8k_seed42.jsonl
```

## Evaluate saved scores

Pairwise metrics and candidate AUROC:

```bash
python scripts/evaluate_scores.py \
  --scores scores/gsm8k_seed42.jsonl
```

If the frozen Best@N subset file is available:

```bash
python scripts/evaluate_scores.py \
  --scores scores/gsm8k_seed42.jsonl \
  --frozen-subsets /path/to/frozen_subsets.json \
  --dataset gsm8k
```

Best@N uses tie-aware expected accuracy: when several candidates share the maximum verifier score, correctness is averaged across the tied maxima.

## Notes on interpretation

SG-LDSV summarizes **local representation transitions**. The released implementation should not be interpreted as establishing that the gate recovers a unique causal notion of reasoning dynamics. The method is unchanged from the paper experiments; the terminology here follows the revised, more conservative manuscript framing.

## License

Apache-2.0. See [LICENSE](LICENSE).

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]