from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch


@dataclass
class ModelSpec:
    name: str
    module: torch.nn.Module
    example_inputs: Tuple[Any, ...]
    example_kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.module.eval()
