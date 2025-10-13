# sanity_run_pte.py
import torch
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program("model.pte")
method = program.load_method("forward")

x = torch.randn(1, 64, 1024)
y, = method.execute([x])  # returns a list; unpack the first
print("ok:", tuple(y.shape), y.dtype)
