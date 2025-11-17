

#include "minixt/instruction.h"
#include "minixt/kernel_registry.h"
#include "minixt/kernels_cpu.h"
#include "minixt/tensor.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>
using namespace minixt;

void test_relu() {
  Tensor x;
  x.shape = {2, 4};
  x.values = {
      -1.0, 0.5, -0.3, 2.0, -4.0, 1.5, -3.0, 5.0,
  };
  print_tensor(x, "input");
  Instruction instr;
  std::vector<Tensor> inputs = {x};
  Tensor y = run_single_op("aten::relu", instr, inputs, {0});
  print_tensor(y, "output");
}
void test_flatten() {
  Tensor x;
  x.shape = {2, 3, 4};
  x.values.resize(24);
  for (size_t i = 0; i < x.values.size(); ++i) {
    x.values[i] = i;
  }
  print_tensor(x, "input_tensor");
  Instruction instr;
  instr.numeric["start_dim"] = 1;
  instr.numeric["end_dim"] = -1;
  Tensor y = run_single_op("aten::flatten", instr, {x}, {0});
  print_tensor(y, "output");
  // printf("%i", y.shape.size());
  // assert(y.shape.size() == 2);
  assert(y.shape[0] == 2);
  assert(y.shape[1] == 12);
}
void test_add_tensor() {
  Tensor a{{3}, {1, 2, 3}};
  Tensor b{{3}, {10, 20, 30}};
  Instruction instr;
  Tensor y = run_single_op("aten::add", instr, {a, b}, {0, 1});
  print_tensor(y, "add output");
}

int main() {
  try {
    register_kernels();
    test_relu();
    test_flatten();
    test_add_tensor();
    //     std::cout << "All tests passed.\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}