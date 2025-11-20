import copy
from pathlib import Path

import torch

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.devtools import generate_etrecord


class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simple conv: 3 -> 32 channels, 3x3 kernel, padding=1
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 3, 224, 224]
        return self.conv(x)


def export_for_backend(
    backend_name: str,
    partitioner,
    out_pte_path: Path,
    out_record_path: Path,
    example_input: torch.Tensor,
) -> None:
    print(f"\n=== Exporting CONV for backend: {backend_name} ===")

    model = ConvModule().eval()

    # 1) torch.export
    exported = torch.export.export(model, (example_input,))

    # 2) Edge + partition
    edge_manager = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=None,
    )

    # 3) Copy for ETRecord before to_executorch (it mutates)
    edge_copy = copy.deepcopy(edge_manager)

    # 4) ExecuTorch program
    et_program = edge_manager.to_executorch()

    # 5) Write .pte
    out_pte_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pte_path.open("wb") as f:
        et_program.write_to_file(f)
    print(f"[OK] wrote {out_pte_path}")

    # 6) Write .etrecord
    generate_etrecord(
        str(out_record_path),
        edge_copy,
        et_program,
        # export_modules=None,
    )
    print(f"[OK] wrote {out_record_path}")


def main():
    # Input: [1, 3, 224, 224]
    example_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    base_dir = Path(__file__).resolve().parent.parent  # examples/backend_bench/
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # XNN
    export_for_backend(
        "xnnpack",
        XnnpackPartitioner(),
        artifacts_dir / "conv_224_xnn.pte",
        artifacts_dir / "conv_224_xnn.etrecord",
        example_input,
    )

    # Vulkan
    export_for_backend(
        "vulkan",
        VulkanPartitioner(),
        artifacts_dir / "conv_224_vulkan.pte",
        artifacts_dir / "conv_224_vulkan.etrecord",
        example_input,
    )


if __name__ == "__main__":
    main()
