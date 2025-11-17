#pragma once
#include <functional>
#include <runtime/core/tensor.h>
#include <runtime/executor/graph.h>
#include <vector>

namespace executorch::runtime::kernel {

using TensorRefVector = std::vector<const core::Tensor *>;
using ParameterMap = std::map<std::string, core::Tensor>;
using KernelFn = std::function<std::vector<core::Tensor>(
    const executor::Instruction &instruction, const TensorRefVector &inputs,
    const ParameterMap &parameters)>;

class KernelRegistry {
public:
  static KernelRegistry &instance();

  void register_kernel(const std::string &op, KernelFn fn);
  const KernelFn *lookup(const std::string &op) const;

private:
  std::unordered_map<std::string, KernelFn> kernels_;
};
} // namespace executorch::runtime::kernel