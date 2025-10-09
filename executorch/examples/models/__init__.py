from __future__ import annotations

from typing import Callable, Dict, List

from .spec import ModelSpec
from . import add, linear, mobilenet_v2

MODEL_REGISTRY: Dict[str, Callable[[str], ModelSpec]] = {
    "add": add.build_model_spec,
    "linear": linear.build_model_spec,
    "mv2": mobilenet_v2.build_model_spec,
}


def list_available_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_model_spec(name: str, *, device: str = "cpu") -> ModelSpec:
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_available_models())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc
    return builder(device)
