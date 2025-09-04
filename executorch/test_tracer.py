from executorch.exir.tracer import dispatch_trace
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, x):
        return x*2 + 1

m = M()
gm = dispatch_trace(m, (torch.randn(3, 3),))
print(gm.graph)
