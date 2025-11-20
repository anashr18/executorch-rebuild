import torch
from pathlib import Path

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)


# ----- 1. Tiny test module: y = x + x -----

class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def export_for_backend(
    backend_name: str,
    partitioner,
    out_path: Path,
    example_input: torch.Tensor,
) -> None:
    print(f"\n=== Exporting for backend: {backend_name} ===")

    model = AddModule().eval()

    # 1) Capture the model with torch.export
    exported_program = torch.export.export(model, (example_input,))
    #    exported_program: "ATen dialect" graph

    # 2) Edge + backend lowering in one call
    et_prog_mgr = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
    ).to_executorch()

    # 3) Serialize to .pte
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        et_prog_mgr.write_to_file(f)

    print(f"[OK] wrote {out_path}")


def main():
    # We fix one canonical shape so both backends see the same workload.
    # example_input = torch.randn(1, 1024, dtype=torch.float32)
    example_input = torch.randn(1, 65536, dtype=torch.float32)

    base_dir = Path(__file__).resolve().parent.parent  # examples/backend_bench/
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # XNNPACK version
    export_for_backend(
        backend_name="xnnpack",
        partitioner=XnnpackPartitioner(),
        out_path=artifacts_dir / "add_xnn.pte",
        example_input=example_input,
    )

    # Vulkan version
    export_for_backend(
        backend_name="vulkan",
        partitioner=VulkanPartitioner(),
        out_path=artifacts_dir / "add_vulkan.pte",
        example_input=example_input,
    )


if __name__ == "__main__":
    main()
