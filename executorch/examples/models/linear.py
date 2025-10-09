from __future__ import annotations

import torch
import torch.nn as nn

from .spec import ModelSpec


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def build_model_spec(device: str = "cpu") -> ModelSpec:
    module = LinearModel().to(device)
    inputs = (torch.randn(1, 4, device=device),)
    return ModelSpec(
        name="linear",
        module=module,
        example_inputs=inputs,
        example_kwargs={},
    )
