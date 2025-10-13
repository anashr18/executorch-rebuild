# export_to_pte_xnn.py
import torch, math
from torch import nn
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        n = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(n + self.eps) * self.weight

m = RMSNorm(1024).eval()
example = torch.randn(1, 64, 1024)

ep = export(m, (example,))
et_prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()

# ✅ Correct: pass a file object
with open("model.pte", "wb") as f:
    et_prog.write_to_file(f)

print("✓ wrote model.pte")
