Goal 1: End-to-End On-Device AI Runtime Pipeline Development

Description:
Develop an end-to-end framework for deploying PyTorch-trained models on edge devices. This includes model export to an intermediate representation (IR), graph optimizations, serialization, and execution through a lightweight C++ runtime supporting multiple backends such as CPU, GPU (Vulkan), and DSP (QNN).

Self-Appraisal / Progress:
Implemented the model export and runtime pipeline from PyTorch to C++ runtime, including IR conversion, serialization, and execution. Began integrating the Vulkan backend, which required building the full C++ runtime from source — a process that initially caused several build and dependency issues, but these were successfully resolved. Currently validating individual operations to ensure stable execution on Vulkan.

Goal 2: Backend Exploration and Performance Optimization

Description:
Explore and benchmark multiple hardware backends (Vulkan GPU, QNN, MediaTek, and open-source runtimes) to evaluate performance, memory efficiency, and deployment feasibility for edge AI workloads.

Self-Appraisal / Progress:
Benchmarked and analyzed various backend options for executing deep learning workloads on-device. Identified key performance bottlenecks, began initial Vulkan integration, and established groundwork for future support of additional backends like QNN.
