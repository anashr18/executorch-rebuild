#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<double> values;
};
// Clone shape + values (relu will overwrite values)
inline Tensor clone_shape(const Tensor &source) {
  Tensor result;
  result.shape = source.shape;
  result.values = source.values;
  return result;
}
inline void print_tensor(const Tensor &t, const std::string &name) {
  std::cout << name << " shape=[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    std::cout << t.shape[i];
    if (i + 1 < t.shape.size())
      std::cout << ", ";
  }
  std::cout << "], values=[";
  for (size_t i = 0; i < t.values.size(); ++i) {
    std::cout << t.values[i];
    if (i + 1 < t.values.size())
      std::cout << ", ";
  }
  std::cout << "]\n";
}