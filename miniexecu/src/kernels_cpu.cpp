#include "minixt/kernels_cpu.h"
#include "minixt/instruction.h"
#include "minixt/kernel_registry.h"
#include "minixt/tensor.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace minixt {
namespace {

const Tensor &require_parameter(const ParameterMap &parameters,
                                const std::string &name) {
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    throw std::runtime_error("Missing parameter '" + name + "'");
  }
  return it->second;
}

std::vector<Tensor> Relu_kernel(const Instruction &instr,
                                const TensorRefVector &inputs,
                                const ParameterMap &param) {
  if (inputs.size() != 1)
    throw std::runtime_error("Relu expects 1 inputs.");
  const Tensor &input = *inputs[0];
  Tensor out = clone_shape(input);
  for (double &value : out.values) {
    if (value < 0.0) {
      value = 0.0;
    }
  }
  return {std::move(out)};
}
std::vector<Tensor> Flatten_kernel(const Instruction &instr,
                                   const TensorRefVector &inputs,
                                   const ParameterMap &param) {
  if (inputs.size() != 1)
    std::runtime_error("Inputs vector should be of size 1");
  const Tensor &input = *inputs[0];
  const auto rank = static_cast<int64_t>(input.shape.size());

  int64_t start_dim =
      static_cast<int64_t>(get_numeric_attr(instr, "start_dim", 0));
  int64_t end_dim =
      static_cast<int64_t>(get_numeric_attr(instr, "end_dim", -1));
  if (start_dim < 0)
    start_dim += rank;
  if (end_dim < 0)
    end_dim += rank;
  if (start_dim < 0 || end_dim >= rank || start_dim > end_dim)
    std::runtime_error("Dims are not valid");

  Tensor out;
  out.shape.reserve(rank - (end_dim - start_dim));

  for (int64_t i = 0; i < start_dim; ++i)
    out.shape.push_back(input.shape[i]);
  int64_t collapsed = 1;
  for (int64_t i = start_dim; i <= end_dim; ++i)
    collapsed *= input.shape[i];
  out.shape.push_back(collapsed);
  for (int64_t i = end_dim + 1; i < rank; ++i)
    out.shape.push_back(input.shape[i]);

  out.values = input.values;
  return {out};
}
std::vector<Tensor> neg_kernel(const Instruction &instr,
                               const TensorRefVector &refs,
                               const ParameterMap &param) {
  if (refs.size() != 1)
    std::runtime_error("Inputs has to be size 1.");
  const Tensor &input = *refs[0];
  Tensor out = clone_shape(input);
  for (int64_t i = 0; i < input.values.size(); ++i)
    out.values[i] = -input.values[i];

  return {out};
}
std::vector<Tensor> mul_scalar_kernel(const Instruction &instr,
                                      const TensorRefVector &refs,
                                      const ParameterMap &param) {
  if (refs.size() != 1)
    std::runtime_error("Inputs has to be size 1.");
  const Tensor &input = *refs[0];
  Tensor out = clone_shape(input);
  double alpha = get_numeric_attr(instr, "alpha", 1.0);
  for (int64_t i = 0; i < input.values.size(); ++i)
    out.values[i] = alpha * input.values[i];
  return {out};
}
std::vector<Tensor> add_kernel(const Instruction &instr,
                               const TensorRefVector &refs,
                               const ParameterMap &param) {
  if (refs.size() != 2)
    std::runtime_error("Inputs has to be size 2.");
  const Tensor &inputA = *refs[0];
  const Tensor &inputB = *refs[1];
  if (inputA.values.size() != inputB.values.size())
    throw std::runtime_error("Size mismatch!");
  Tensor out = clone_shape(inputA);
  for (int64_t i = 0; i < inputA.values.size(); ++i)
    out.values[i] = inputA.values[i] + inputB.values[i];
  return {out};
}
std::vector<Tensor> Liner_kernel(const Instruction &instr,
                                 const TensorRefVector &refs,
                                 const ParameterMap &param) {
  const Tensor &input = *refs[0];
  if (input.shape.size() != 2)
    throw std::runtime_error("aten::Linear input must be of shape 2D");

  const auto &strings = instr.strings;
  const std::string &weight_name = strings.at("weight");
  const Tensor &weight = require_parameter(param, weight_name);
  if (weight.shape.size() != 2) {
    throw std::runtime_error(
        "aten::linear: weight must be 2D [out_features, in_features]");
  }

  auto bias_it = strings.find("bias");
  const Tensor *bias = nullptr;
  if (bias_it != strings.end())
    bias = &require_parameter(param, bias_it->second);

  const int64_t batch = input.shape[0];
  const int64_t in_features = input.shape[1];
  const int64_t out_features = weight.shape[0];

  if (weight.shape[1] != in_features) {
    throw std::runtime_error(
        "aten::linear: in_features mismatch between input and weight");
  }

  if (bias && bias->shape.size() != 1) {
    throw std::runtime_error("aten::linear: bias must be 1D [out_features]");
  }
  if (bias && bias->shape[0] != out_features) {
    throw std::runtime_error("aten::linear: bias length mismatch");
  }
}
} // namespace
void register_kernels() {
  auto &registry = KernelRegistry::instance();
  registry.register_kernel("aten::relu", &Relu_kernel);
  registry.register_kernel("aten::flatten", &Flatten_kernel);
  registry.register_kernel("aten::neg", &neg_kernel);
  registry.register_kernel("aten::add", &add_kernel);
}
} // namespace minixt