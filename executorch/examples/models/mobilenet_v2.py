from __future__ import annotations

import torch
from torchvision.models import mobilenet_v2

from .spec import ModelSpec


def build_model_spec(device: str = "cpu") -> ModelSpec:
    module = mobilenet_v2(weights=None).to(device)
    inputs = (torch.randn(1, 3, 224, 224, device=device),)
    return ModelSpec(
        name="mv2",
        module=module,
        example_inputs=inputs,
        example_kwargs={},
    )
