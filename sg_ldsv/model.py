from __future__ import annotations

import torch
from torch import nn


def first_last_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,T]")
    first = mask.float().argmax(dim=1)
    last_from_end = torch.flip(mask, dims=[1]).float().argmax(dim=1)
    last = mask.shape[1] - 1 - last_from_end
    valid = mask.any(dim=1)
    return first, last, valid


def gather_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _, last, valid = first_last_indices(mask)
    if not bool(valid.all()):
        raise RuntimeError("empty response sequence reached verifier head")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, last]


class EndpointMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SGLDSVHead(nn.Module):
    """Exact SG-LDSV verifier head used by the paper experiments."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 128, bias=False)
        self.proj_ln = nn.LayerNorm(128)
        self.endpoint = EndpointMLP(d)
        self.lambda_local = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.phi = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
        self.gate = nn.Linear(128, 1)
        self.rho = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def projected(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj_ln(self.proj(hidden))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B,T,d]")
        if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,T] aligned with hidden")

        z = self.projected(hidden)
        delta = z[:, 1:] - z[:, :-1]
        transition_mask = mask[:, 1:] & mask[:, :-1]

        content = self.phi(delta)
        gate_values = torch.sigmoid(self.gate(z[:, :-1])).squeeze(-1)
        weights = gate_values * transition_mask.float()
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        mean = (weights.unsqueeze(-1) * content).sum(dim=1) / denominator
        centered = content - mean.unsqueeze(1)
        variance = (weights.unsqueeze(-1) * centered.square()).sum(dim=1) / denominator
        std = torch.sqrt(variance + 1e-6)

        local_score = self.rho(torch.cat([mean, std], dim=-1)).squeeze(-1)
        has_transition = transition_mask.any(dim=1)
        local_score = torch.where(has_transition, local_score, torch.zeros_like(local_score))

        endpoint_score = self.endpoint(gather_last(hidden, mask)).squeeze(-1)
        score = endpoint_score + self.lambda_local * local_score

        return {
            "score": score,
            "endpoint_score": endpoint_score,
            "local_score": local_score,
            "lambda": self.lambda_local.expand_as(score),
            "gate_values": gate_values,
            "transition_mask": transition_mask,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]