#pragma once
#include <ostream>
#include <string>
#include <vector>

// #include "runtime/core/tensor.h"
#include "runtime/executor/program.h"

namespace executorch::runtime::executor {

class Executor {
public:
  static Executor load(const std::string &path);

  void summarize(std::ostream &out) const;
  void run(std::ostream &out) const;

private:
  explicit Executor(Program program);

  std::vector<core::Tensor> execute_graph() const;

  Program program_;
};

} // namespace executorch::runtime::executor
