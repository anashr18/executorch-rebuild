from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass
class EdgeProgram:
    module_spec: Mapping[str, Any]
    execution: Mapping[str, Any]
    metadata: Mapping[str, Any]
    format_version: str = "0.0.1"

    def to_executorch(self) -> "ExecuTorchProgram":
        payload = {
            "module": dict(self.module_spec),
            "execution": dict(self.execution),
            "metadata": dict(self.metadata),
        }
        metadata = {
            "format": "executorch.json",
            "edge_version": self.format_version,
        }
        return ExecuTorchProgram(metadata=metadata, payload=payload)

@dataclass
class ExecuTorchProgram:
    metadata: Mapping[str, Any]
    payload: Mapping[str, Any]

    @property
    def buffer(self) -> bytes:
        packed = {"metadata": self.metadata, "payload": self.payload}
        return json.dumps(packed, sort_keys=True, indent=2).encode("utf-8")

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.write_bytes(self.buffer)
        return target
