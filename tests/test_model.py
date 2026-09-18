[Reading 22 lines from start (total: 22 lines, 0 remaining)]

import torch

from sg_ldsv.model import SGLDSVHead


def test_forward_shapes_and_finiteness():
    torch.manual_seed(0)
    head = SGLDSVHead(64)
    hidden = torch.randn(3, 6, 64)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    out = head(hidden, mask)
    assert out["score"].shape == (3,)
    assert out["gate_values"].shape == (3, 5)
    assert torch.isfinite(out["score"]).all()
    assert out["local_score"][2].item() == 0.0

[executed on device: autodl-container-ceda25je6k-ffe84779 (6dad3e05-41ae-4cc2-b58f-d95702e6e179)]