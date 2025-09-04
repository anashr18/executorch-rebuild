import torch
from typing import Generator
from contextlib import contextmanager

@contextmanager
def no_dispatch() -> Generator[None, None, None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard
