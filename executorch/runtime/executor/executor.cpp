#include "runtime/executor/executor.h"
#include "runtime/core/tensor.h"
#include "runtime/kernel/registry.h"
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace executorch::runtime::executor {

using runtime::kernel::KernelRegistry;
using runtime::kernel::ParameterMap;
using runtime::kernel::TensorRefVector;

Executor::Executor(Program program) : program_(std::move(program)) {}

Executor Executor::load(const std::string &path) {
  return Executor{Program::load_from_file(path)};
}

void Executor::summarize(std::ostream &out) const {
  out << "Module: " << program_.qualified_name << "\n";
  out << "Capture device: " << program_.capture_device << "\n";
  out << "Parameters captured: " << program_.state_dict.size() << "\n";
  out << "Graph inputs: " << program_.graph.input_names.size() << "\n";
  out << "Graph instructions: " << program_.graph.instructions.size() << "\n";
}

std::vector<core::Tensor> Executor::execute_graph() const {
  std::unordered_map<std::string, core::Tensor> values;

  ParameterMap parameters;
  for (const auto &[name, tensor_data] : program_.state_dict) {
    parameters.emplace(name, tensor_data.to_tensor());
  }

  const auto &input_names = program_.graph.input_names;
  const auto &input_array = program_.inputs.as_array();
  if (input_array.size() != input_names.size()) {
    throw std::runtime_error(
        "Executor: mismatch between graph inputs and serialized inputs");
  }

  for (std::size_t i = 0; i < input_names.size(); ++i) {
    values.emplace(input_names[i], core::tensor_from_json(input_array[i]));
  }

  const auto &registry = KernelRegistry::instance();

  for (const auto &instruction : program_.graph.instructions) {
    TensorRefVector instruction_inputs;
    instruction_inputs.reserve(instruction.inputs.size());
    for (const auto &name : instruction.inputs) {
      const auto value_it = values.find(name);
      if (value_it == values.end()) {
        throw std::runtime_error("Executor: missing input '" + name +
                                 "' for instruction '" + instruction.name +
                                 "'");
      }
      instruction_inputs.push_back(&value_it->second);
    }

    const auto *kernel_fn = registry.lookup(instruction.op);
    if (!kernel_fn) {
      throw std::runtime_error("Executor: no kernel registered for op '" +
                               instruction.op + "'");
    }

    auto results = (*kernel_fn)(instruction, instruction_inputs, parameters);
    if (results.size() != instruction.outputs.size()) {
      throw std::runtime_error("Executor: kernel for '" + instruction.op +
                               "' returned unexpected number of tensors");
    }

    for (std::size_t index = 0; index < instruction.outputs.size(); ++index) {
      values[instruction.outputs[index]] = std::move(results[index]);
    }
  }

  std::vector<core::Tensor> outputs;
  outputs.reserve(program_.graph.output_names.size());
  for (const auto &name : program_.graph.output_names) {
    const auto value_it = values.find(name);
    if (value_it == values.end()) {
      throw std::runtime_error("Executor: missing output '" + name + "'");
    }
    outputs.push_back(value_it->second);
  }

  return outputs;
}

void Executor::run(std::ostream &out) const {
  const auto outputs = execute_graph();
  out << "Runtime outputs:\n";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    out << "  output[" << index
        << "]: " << core::tensor_to_json(outputs[index]).dump(2) << "\n";
  }
  out << "Captured reference outputs:\n";
  out << program_.outputs.dump(2) << "\n";
}

} // namespace executorch::runtime::executor
