
#include "runtime/core/json.h"
#include <cstddef>
#include <cstdint>
#include <runtime/core/tensor.h>
#include <stdexcept>

namespace executorch::runtime::core {
std::size_t Tensor::numel() const {
  std::size_t elements = 1;
  for (int64_t dim : shape) {
    elements *= static_cast<std::size_t>(dim);
  }
  return elements;
}
Tensor tensor_from_json(const JsonValue &value) {
  const auto &object = value.as_object();

  const auto type_it = object.find("type");
  if (type_it == object.end() || type_it->second.as_string() != "tensor") {
    throw std::runtime_error("Tensor_from_josn:: expected tensor from json");
  }
  Tensor tensor;
  tensor.dtype = object.at("dtype").as_string();

  const auto &shape_array = object.at("shape").as_array();
  tensor.shape.reserve(shape_array.size());
  for (const auto &entry : shape_array) {
    tensor.shape.push_back(static_cast<int64_t>(entry.as_number()));
  }
  const auto &values_array = object.at("values").as_array();
  tensor.values.reserve(values_array.size());
  for (const auto &entry : values_array) {
    tensor.values.push_back(entry.as_number());
  }
  if (!tensor.shape.empty() && tensor.numel() != tensor.values.size()) {
    throw std::runtime_error(
        "Tensor from json:: Shape does not match vfalue count");
  }
  return tensor;
}
JsonValue tensor_to_json(const Tensor &tensor) {
  JsonValue::object_type object;
  object.emplace("type",
                 JsonValue(JsonValue::value_type{std::string("tensor")}));
  object.emplace("dtype", JsonValue(JsonValue::value_type{tensor.dtype}));

  JsonValue::array_type shape;
  shape.reserve(tensor.shape.size());
  for (int64_t dim : tensor.shape) {
    shape.emplace_back(
        JsonValue(JsonValue::value_type{static_cast<double>(dim)}));
  }
  object.emplace("shape", JsonValue(shape));

  JsonValue::array_type values;
  values.reserve(tensor.values.size());
  for (double value : tensor.values) {
    values.emplace_back(JsonValue(JsonValue::value_type{value}));
  }
  object.emplace("values", JsonValue(values));

  return JsonValue(object);
}
} // namespace executorch::runtime::core