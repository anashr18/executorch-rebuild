#pragma once

#include <ostream>
#include <string>

#include "runtime/executor/program.h"

namespace executorch::runtime::executor {

class Executor {
 public:
  static Executor load(const std::string& path);

  void summarize(std::ostream& out) const;
  void run(std::ostream& out) const;

 private:
  explicit Executor(Program program);

  Program program_;
};

}  // namespace executorch::runtime::executor
