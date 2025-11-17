
#pragma once
#include "runtime/core/json.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch::runtime::core {
struct Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<double> values;
  std::size_t numel() const;
};
Tensor tensor_from_json(const JsonValue &value);
JsonValue tensor_to_json(const Tensor &value);
} // namespace executorch::runtime::core