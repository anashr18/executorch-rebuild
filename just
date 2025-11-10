Research Update — ExecuTorch Pipeline and Vulkan Runtime Integration

This document summarizes ongoing research and technical exploration around the ExecuTorch runtime architecture and Vulkan backend enablement.
The goal is to build a deeper understanding of the end-to-end pipeline—from model tracing and export to runtime kernel execution—and to enable Vulkan GPU acceleration on edge devices such as Android.

1. ExecuTorch Pipeline Analysis

The initial phase focused on understanding the overall pipeline of ExecuTorch and how computation flows from PyTorch models to C++ runtime execution.
This analysis formed the foundation for debugging backend integration issues and for identifying Vulkan runtime gaps.

1.1 Conceptual Flow
[ PyTorch nn.Module ]
          │
          ▼
[ FX Graph via torch.fx.symbolic_trace ]
          │
          ▼
[ ATen IR (Intermediate Representation) ]
          │
          ▼
[ Edge IR – optimized for mobile/edge ]
          │
          ▼
[ Backend Partitioning (Vulkan / XNNPACK) ]
          │
          ▼
[ ExecuTorch Runtime Execution (.pte) ]


Each layer serves a distinct purpose:

FX Graph: Converts the dynamic Python model into a static computation graph using symbolic tracing.

ATen IR: Expresses all computation as ATen operators, removing Python control flow.

Edge IR: A lightweight, deterministic IR tailored for mobile and embedded execution.

Backend Partitioning: Assigns supported subgraphs to backends such as Vulkan or XNNPACK.

Runtime Execution: Executes serialized .pte graphs on the C++ runtime with dynamic kernel dispatch.

1.2 Tracing and Graph Generation

Using torch.fx.symbolic_trace, models are captured as functional graphs.
Each operation becomes a node with explicit data dependencies, allowing static analysis and transformation.

from torch.fx import symbolic_trace
gm = symbolic_trace(model)
print(gm.graph)


The output defines how the model computes rather than how Python executes it.

1.3 IR Transformations

After FX tracing, the model transitions through successive intermediate representations:

Stage	Description
ATen IR	Pure tensor operations independent of Python or control flow
Edge IR	Pruned and deterministic representation optimized for on-device runtime
Backend Partitioning	Delegates supported operator subgraphs to target backends

These transformations ensure a portable, hardware-agnostic representation before export.

1.4 Kernel Dispatching

At runtime, ExecuTorch loads the .pte graph and executes nodes in order:

for (auto& node : execution_order) {
    auto backend = node.assigned_backend;
    auto kernel_fn = registry.lookup(node.op, backend);
    kernel_fn(node.inputs, node.outputs);
}


The Kernel Registry maps operator names to implementations for each backend.
This allows dynamic dispatch without embedding backend logic into the runtime.

CPU kernels: ATen-style C++ functions

Vulkan kernels: Compute shaders using GPU buffers

XNNPACK kernels: SIMD-optimized CPU kernels

This registry-based execution model is the backbone of ExecuTorch’s modular design.

1.5 Quantization and Fusion

Quantization and fusion occur before backend lowering—typically between ATen IR and Edge IR.

 FX Graph → ATen IR → Edge IR
      │         │         │
      └── Fusion ─┘   └── Quantization ─┘
                      ↓
                Backend Partitioning


Fusion: Combines compatible ops (e.g., conv2d + relu) for fewer kernel calls.

Quantization: Converts floating-point operations to INT8/FP16 with scaling metadata.

These passes ensure the exported model graph is backend-optimized and numerically consistent.

2. Vulkan Runtime Build
2.1 Objective

Integrate and validate the Vulkan backend in ExecuTorch for GPU-accelerated inference on Android and embedded edge systems.

2.2 Environment and Setup

Base runtime: ExecuTorch built from source

Build system: CMake with Vulkan backend flag enabled

Target platforms: Android (primary) and Linux (for debugging)

Dependencies: Vulkan SDK via Android NDK, shader tools (glslangValidator, shaderc), and driver validation layers

2.3 Build Summary

During initial builds, Vulkan components were skipped due to missing compile-time definitions such as USE_VULKAN.
Because these flags propagate across multiple submodules, any missing entry prevents backend compilation and registry linkage.

After correcting the flag propagation, Vulkan sources compiled and linked successfully, confirming the backend inclusion path was intact.

2.4 Observations

Vulkan backend integration is optional and not enabled by default.

Operator coverage is limited; unsupported ops fall back to CPU.

Vulkan memory and shader layers exist but require deeper runtime integration.

Backend registration occurs through macros like REGISTER_BACKEND(vulkan, VulkanBackend) that bind Vulkan kernels to the global registry.

2.5 Key Points

Build configuration sensitivity: Missing flags disable entire backends.

Hybrid execution: Vulkan handles supported ops; CPU/XNNPACK manage the rest.

Registration consistency: Macros must align with kernel registry expectations.

Next focus: Extend Vulkan operator coverage and improve build automation.

3. Vulkan Support Limitations
Area	Limitation	Impact
Operator coverage	Limited supported ops (add, relu, conv2d)	CPU fallback for missing ops
Memory management	Partial Vulkan allocator integration	Buffer reuse inefficiency
Shader toolchain	Version mismatches possible between SPIR-V and drivers	Validation errors during runtime
Build configuration	Optional inclusion through compile flags	Backend often excluded from multi-target builds
Dispatch registry	Partial linkage of Vulkan operators	Missing runtime dispatch for some ops

Vulkan execution is functional but incomplete. Current builds demonstrate hybrid CPU + Vulkan inference rather than full GPU pipelines.

4. Technical Insights

Backend Modularity – ExecuTorch cleanly separates tracing, export, and runtime. This modularity enables independent backend work.

Dynamic Dispatch – Kernels are bound at runtime through a registry, allowing small binaries and flexible backend loading.

Build Sensitivity – CMake and macro configuration must be precisely managed for backend inclusion.

Unified Operator Schema – All backends must adhere to the same ATen operator signatures.

Incomplete Vulkan Coverage – Operator registration and planner integration remain partial.

Tracing vs. Execution Separation – Tracing produces the graph; execution only consumes it. Understanding this distinction simplifies backend debugging.

5. Current Focus and Next Steps

Vulkan Stabilization

Ensure flag propagation across CMake targets.

Automate build checks for Vulkan dependency presence.

Validate kernel loading on Android hardware.

Registry Expansion

Extend Vulkan backend coverage for common arithmetic and convolution ops.

Verify end-to-end execution through Vulkan subgraphs.

Runtime Profiling

Measure kernel dispatch latency and memory usage.

Compare Vulkan vs. CPU performance on representative models.

Cross-Platform Validation

Run .pte exports on Android devices and Linux debug builds.

Check shader portability and driver behavior.

Source Exploration

Continue reviewing ExecuTorch’s backend source to identify missing integration points.

Propose patches for consistent Vulkan inclusion.

6. Summary

The research established a clear understanding of ExecuTorch’s runtime pipeline and successfully integrated Vulkan backend compilation with partial execution capability.
Vulkan currently operates in hybrid mode with CPU fallback but forms the foundation for complete GPU acceleration in future releases.

The refined grasp of FX tracing, IR transformations, backend partitioning, and kernel dispatch strengthens our ability to debug and extend runtimes for edge inference.
Next objectives include stabilizing Vulkan support on Android, expanding operator coverage, and validating performance improvements through profiling and multi-backend execution tests.
