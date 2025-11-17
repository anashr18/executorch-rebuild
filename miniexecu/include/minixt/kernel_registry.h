#pragma once

#include "instruction.h"
#include "tensor.h"
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace minixt {

using TensorRefVector = std::vector<const Tensor *>;
using ParameterMap = std::unordered_map<std::string, Tensor>;

using KernelFn = std::vector<Tensor> (*)(const Instruction &,
                                         const TensorRefVector &,
                                         const ParameterMap &);

class KernelRegistry {
public:
  static KernelRegistry &instance() {
    static KernelRegistry inst;
    return inst;
  }
  void register_kernel(const std::string &name, KernelFn fn) {
    auto it = register_.find(name);
    if (it != register_.end())
      throw std::runtime_error(
          "A kernel already registered with the op name: " + name);
    register_[name] = fn;
  }
  KernelFn get_kernel(const std::string &name) {
    auto it = register_.find(name);
    if (it == register_.end())
      throw std::runtime_error("No kernel registered with the op: " + name);
    return it->second;
  }

private:
  std::unordered_map<std::string, KernelFn> register_;
};

inline Tensor run_single_op(const std::string &op_name,
                            const Instruction &instr,
                            const std::vector<Tensor> &all_inputs,
                            const std::vector<size_t> &input_indices,
                            const ParameterMap &param = {}) {
  TensorRefVector refs;
  refs.reserve(input_indices.size());
  for (size_t idx : input_indices) {
    if (idx >= all_inputs.size())
      throw std::runtime_error("Input index out of range");
    refs.push_back(&all_inputs[idx]);
  }
  KernelFn fn = KernelRegistry::instance().get_kernel(op_name);
  std::vector<Tensor> outputs = fn(instr, refs, param);

  if (outputs.size() != 1)
    throw std::runtime_error("Output should be of one size now.");
  return outputs[0];
}
} // namespace minixt