import torch, math, copy
from typing import List, Optional, Tuple, Union
from executorch.exir.schema import TensorShapeDynamism


class TensorSpec:
    def __init__(self, dtype: torch.dtype, shape: torch.Size, const: bool = False):
        self.scalar_type = dtype
        self.shape = list(shape)
        self.const = const
        self.lifetime = [None, None]
        self.mem_id = None
        self.mem_offset = None
        self.shape_dynamism = TensorShapeDynamism.STATIC  # placeholder

    def debug(self) -> str:
        return f"TensorSpec(dtype={self.scalar_type}, shape={self.shape}, const={self.const})"
