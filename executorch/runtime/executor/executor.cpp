#include "runtime/executor/executor.h"

#include <utility>

namespace executorch::runtime::executor {

Executor::Executor(Program program) : program_(std::move(program)) {}

Executor Executor::load(const std::string& path) {
  return Executor{Program::load_from_file(path)};
}

void Executor::summarize(std::ostream& out) const {
  out << "Module: " << program_.qualified_name << "\n";
  out << "Capture device: " << program_.capture_device << "\n";
  out << "Parameters captured: " << program_.state_dict.size() << "\n";
}

void Executor::run(std::ostream& out) const {
  out << "Replaying captured outputs:\n";
  out << program_.outputs.dump(2) << "\n";
}

}  // namespace executorch::runtime::executor
