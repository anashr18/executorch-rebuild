
#include "runtime/core/json.h"
#include <cstddef>
#include <cstdint>
#include <runtime/core/tesnor.h>

namespace executorch::runtime::core {
std::size_t Tensor::numel() const {
  std::size_t elements = 1;
  for (int64_t dim : shape) {
    elements *= static_cast<std::size_t>(dim);
  }
  return elements;
}
Tensor json_to_tensor(const JsonValue &value) {}
} // namespace executorch::runtime::core