from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

_PRIMITIVES = (int, float, bool, type(None), str)


def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    detached = tensor.detach().cpu()
    return {
        "type": "tensor",
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "values": detached.flatten().tolist(),
    }


def serialize_outputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return serialize_tensor(value)
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_outputs(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): serialize_outputs(v) for k, v in value.items()}
    return repr(value)


def serialize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    return {key: serialize_tensor(tensor) for key, tensor in state_dict.items()}
