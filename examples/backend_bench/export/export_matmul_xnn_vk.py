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
from executorch.devtools import generate_etrecord  # NEW


class MatmulModule(torch.nn.Module):
    def __init__(self, in_features: int = 1024, out_features: int = 1024):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, in_features]
        return x @ self.weight  # [1, out_features]


def export_for_backend(
    backend_name: str,
    partitioner,
    out_pte_path: Path,
    out_record_path: Path,
    example_input: torch.Tensor,
) -> None:
    print(f"\n=== Exporting MATMUL for backend: {backend_name} ===")

    model = MatmulModule().eval()

    # 1) torch.export
    exported = torch.export.export(model, (example_input,))

    # 2) Edge dialect + transforms + partition
    edge_manager = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=None,  # or EdgeCompileConfig if you want; None is fine here
    )

    # 3) Deep copy BEFORE to_executorch for ETRecord
    edge_manager_copy = copy.deepcopy(edge_manager)

    # 4) ExecuTorch program
    et_program = edge_manager.to_executorch()

    # 5) Write .pte
    out_pte_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pte_path.open("wb") as f:
        et_program.write_to_file(f)
    print(f"[OK] wrote {out_pte_path}")

    # 6) Write ETRecord (debug/profiling metadata)
    generate_etrecord(
        str(out_record_path),
        edge_manager_copy,  # EdgeProgramManager (copied)
        et_program,         # ExecutorchProgramManager
        # export_modules=None,
    )
    print(f"[OK] wrote {out_record_path}")


def main():
    example_input = torch.randn(1, 1024, dtype=torch.float32)

    base_dir = Path(__file__).resolve().parent.parent  # examples/backend_bench/
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # XNN
    export_for_backend(
        "xnnpack",
        XnnpackPartitioner(),
        artifacts_dir / "matmul_xnn.pte",
        artifacts_dir / "matmul_xnn.etrecord",
        example_input,
    )

    # Vulkan
    export_for_backend(
        "vulkan",
        VulkanPartitioner(),
        artifacts_dir / "matmul_vulkan.pte",
        artifacts_dir / "matmul_vulkan.etrecord",
        example_input,
    )


if __name__ == "__main__":
    main()
