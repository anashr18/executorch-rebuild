// #include "runtime/core/json.h"
#include "runtime/executor/program.h"
#include "runtime/core/json.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace executorch::runtime::executor {

namespace {

using executorch::runtime::core::JsonValue;
std::string read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::vector<std::string> parse_string_list(const JsonValue &value) {
  const auto &array = value.as_array();
  std::vector<std::string> result;
  result.reserve(array.size());
  for (const auto &entry : array) {
    result.push_back(entry.as_string());
  }
  return result;
}

Instruction parse_instruction(const JsonValue &value) {
  const auto &object = value.as_object();

  Instruction instruction;
  instruction.name = object.at("name").as_string();
  instruction.op = object.at("op").as_string();
  instruction.inputs = parse_string_list(object.at("inputs"));
  instruction.outputs = parse_string_list(object.at("outputs"));

  const auto attr_it = object.find("attributes");
  if (attr_it != object.end()) {
    const auto &attributes = attr_it->second.as_object();

    const auto numeric_it = attributes.find("numeric");
    if (numeric_it != attributes.end()) {
      for (const auto &[key, raw_value] : numeric_it->second.as_object()) {
        instruction.attributes.numeric.emplace(key, raw_value.as_number());
      }
    }
    const auto string_it = attributes.find("string");
    if (string_it != attributes.end()) {
      for (const auto &[key, raw_value] : string_it->second.as_object()) {
        instruction.attributes.strings.emplace(key, raw_value.as_string());
      }
    }
  }
  return instruction;
}
ProgramGraph parse_graph(const JsonValue &value) {
  const auto &object = value.as_object();
  ProgramGraph graph;
  graph.input_names = parse_string_list(object.at("inputs"));
  graph.output_names = parse_string_list(object.at("outputs"));

  const auto &instruction_array = object.at("instructions").as_array();
  graph.instructions.reserve(instruction_array.size());
  for (const auto &entry : instruction_array) {
    graph.instructions.emplace_back(parse_instruction(entry));
  }
  return graph;
}
TensorData parse_tensor_data(const JsonValue &value) {
  const auto &object = value.as_object();

  TensorData data;
  data.dtype = object.at("dtype").as_string();

  const auto &shape_values = object.at("shape").as_array();
  data.shape.reserve(shape_values.size());
  for (const auto &dim : shape_values) {
    data.shape.push_back(static_cast<int64_t>(std::llround(dim.as_number())));
  }

  const auto &value_array = object.at("values").as_array();
  data.values.reserve(value_array.size());
  for (const auto &raw : value_array) {
    data.values.push_back(raw.as_number());
  }

  return data;
}
} // namespace
core::Tensor TensorData::to_tensor() const {
  core::Tensor tensor;
  tensor.dtype = dtype;
  tensor.shape = shape;
  tensor.values = values;
  return tensor;
}

Program Program::load_from_file(const std::string &path) {
  const auto contents = read_file(path);
  auto root = core::parse_json(contents);

  const auto &root_object = root.as_object();
  const auto &payload = root_object.at("payload").as_object();
  const auto &module = payload.at("module").as_object();
  const auto &execution = payload.at("execution").as_object();
  const auto &capture_metadata = payload.at("metadata").as_object();

  Program program;
  program.qualified_name = module.at("qualified_name").as_string();

  const auto capture_it = capture_metadata.find("capture_device");
  program.capture_device = (capture_it != capture_metadata.end())
                               ? capture_it->second.as_string()
                               : "unknown";

  const auto &state_dict = module.at("state_dict").as_object();
  for (const auto &[name, tensor_json] : state_dict) {
    program.state_dict.emplace(name, parse_tensor_data(tensor_json));
  }

  program.inputs = execution.at("inputs");
  program.kwargs = execution.at("kwargs");
  program.outputs = execution.at("outputs");

  const auto graph_it = execution.find("graph");
  if (graph_it == execution.end()) {
    throw std::runtime_error("Program file missing execution graph");
  }
  program.graph = parse_graph(graph_it->second);

  return program;
}

} // namespace executorch::runtime::executor