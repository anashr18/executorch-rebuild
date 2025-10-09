#include "runtime/executor/program.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace executorch::runtime::executor {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open program file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

int64_t to_int64(const core::JsonValue& value) {
  return static_cast<int64_t>(std::llround(value.as_number()));
}

}  // namespace

Program Program::load_from_file(const std::string& path) {
  const auto contents = read_file(path);
  auto root = core::parse_json(contents);

  const auto& root_object = root.as_object();
  const auto& payload = root_object.at("payload").as_object();
  const auto& module = payload.at("module").as_object();
  const auto& execution = payload.at("execution").as_object();
  const auto& capture_metadata = payload.at("metadata").as_object();

  Program program;
  program.qualified_name = module.at("qualified_name").as_string();

  const auto capture_it = capture_metadata.find("capture_device");
  if (capture_it != capture_metadata.end()) {
    program.capture_device = capture_it->second.as_string();
  } else {
    program.capture_device = "unknown";
  }

  const auto& state_dict = module.at("state_dict").as_object();
  for (const auto& [name, tensor_json] : state_dict) {
    const auto& tensor_object = tensor_json.as_object();
    TensorData tensor;
    tensor.dtype = tensor_object.at("dtype").as_string();

    const auto& shape_values = tensor_object.at("shape").as_array();
    tensor.shape.reserve(shape_values.size());
    for (const auto& dim : shape_values) {
      tensor.shape.push_back(to_int64(dim));
    }

    const auto& value_list = tensor_object.at("values").as_array();
    tensor.values.reserve(value_list.size());
    for (const auto& raw : value_list) {
      tensor.values.push_back(raw.as_number());
    }

    program.state_dict.emplace(name, std::move(tensor));
  }

  program.inputs = execution.at("inputs");
  program.kwargs = execution.at("kwargs");
  program.outputs = execution.at("outputs");

  return program;
}

}  // namespace executorch::runtime::executor
