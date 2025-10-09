from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch


def ensure_tuple(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def move_sequence_to_device(sequence: Any, device: str) -> Tuple[Any, ...]:
    sequence = ensure_tuple(sequence)
    return tuple(move_to_device(v, device) for v in sequence)


def move_mapping_to_device(mapping: Mapping[str, Any], device: str) -> Mapping[str, Any]:
    return {k: move_to_device(v, device) for k, v in mapping.items()}


def detach_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(detach_to_cpu(v) for v in value)
    if isinstance(value, Mapping):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    return value
