from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AllocationDetails:
    memory_id: int
    memory_offset: int


class TensorShapeDynamism(IntEnum):
    STATIC = 0
    DYNAMIC_BOUND = 1
    DYNAMIC_UNBOUND = 2


@dataclass
class Tensor:
    scalar_type: int
    storage_offset: int
    sizes: List[int]
    dim_order: List[bytes]
    requires_grad: bool
    layout: int
    constant_buffer_idx: int
    allocation_info: Optional[AllocationDetails]
    shape_dynamism: TensorShapeDynamism
