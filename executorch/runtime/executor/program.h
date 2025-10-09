#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "runtime/core/json.h"

namespace executorch::runtime::executor {

struct TensorData {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<double> values;
};

struct Program {
  std::string qualified_name;
  std::string capture_device;
  std::map<std::string, TensorData> state_dict;
  core::JsonValue inputs;
  core::JsonValue kwargs;
  core::JsonValue outputs;

  static Program load_from_file(const std::string& path);
};

}  // namespace executorch::runtime::executor
