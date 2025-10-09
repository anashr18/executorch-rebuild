from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

from .program import EdgeProgram
from .serialization import serialize_outputs, serialize_state_dict
from .utils import (
    detach_to_cpu,
    ensure_tuple,
    move_mapping_to_device,
    move_sequence_to_device,
)


@dataclass
class CapturedProgram:
    module: torch.nn.Module
    inputs: Tuple[Any, ...]
    kwargs: Mapping[str, Any]
    outputs: Any
    device: str

    def to_edge(self) -> EdgeProgram:
        module_spec = {
            "qualified_name": (
                f"{self.module.__class__.__module__}."
                f"{self.module.__class__.__qualname__}"
            ),
            "state_dict": serialize_state_dict(self.module.state_dict()),
        }
        execution = {
            "inputs": serialize_outputs(self.inputs),
            "kwargs": {k: serialize_outputs(v) for k, v in self.kwargs.items()},
            "outputs": serialize_outputs(self.outputs),
        }
        metadata = {"capture_device": self.device}
        return EdgeProgram(
            module_spec=module_spec,
            execution=execution,
            metadata=metadata,
        )


def capture(
    module: torch.nn.Module,
    example_inputs: Any,
    example_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> CapturedProgram:
    if example_inputs is None:
        raise ValueError("example_inputs must be provided")

    example_kwargs = dict(example_kwargs or {})
    inputs = ensure_tuple(example_inputs)

    working_module = module
    target_device = device or "cpu"

    if device:
        working_module = working_module.to(device)

    inputs_on_device = move_sequence_to_device(inputs, target_device)
    kwargs_on_device = move_mapping_to_device(example_kwargs, target_device)

    if eval_mode:
        working_module = working_module.eval()

    with torch.no_grad():
        outputs = working_module(*inputs_on_device, **kwargs_on_device)

    captured_inputs = detach_to_cpu(inputs_on_device)
    captured_kwargs = detach_to_cpu(kwargs_on_device)
    captured_outputs = detach_to_cpu(outputs)

    working_module = working_module.to("cpu")

    return CapturedProgram(
        module=working_module,
        inputs=captured_inputs,
        kwargs=captured_kwargs,
        outputs=captured_outputs,
        device=target_device,
    )
