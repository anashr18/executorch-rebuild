from __future__ import annotations

import torch
import torch.nn as nn

from .spec import ModelSpec


class AddModel(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b



def build_model_spec(device: str = "cpu") -> ModelSpec:
    module = AddModel().to(device)
    inputs = (
        torch.ones(1, device=device),
        torch.full((1,), 2.0, device=device),
    )
    return ModelSpec(
        name="add",
        module=module,
        example_inputs=inputs,
        example_kwargs={},
    )
