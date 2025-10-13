# scripts/export_vulkan.py
import torch
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

# demo module; replace with your RMSNorm/Attention toy nets
class Add(torch.nn.Module):
    def forward(self, x, y): return x + y

aten = export(Add(), (torch.ones(1), torch.ones(1)))
edge = to_edge(aten)
edge = edge.to_backend(VulkanPartitioner())      # <-- Vulkan lowering
et_prog = edge.to_executorch()

with open("model_vk.pte", "wb") as f:
    f.write(et_prog.buffer)
