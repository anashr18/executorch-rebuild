#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/executor/executor.h"

using executorch::runtime::executor::Executor;

namespace {

struct Options {
  std::string model_path;
  bool verbose = false;
};

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--model_path" || arg == "-m") && i + 1 < argc) {
      options.model_path = argv[++i];
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: executor_runner --model_path <path.ff> [--verbose]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (options.model_path.empty()) {
    throw std::runtime_error("--model_path is required");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_args(argc, argv);
    auto executor = Executor::load(options.model_path);
    executor.summarize(std::cout);
    executor.run(std::cout);
  } catch (const std::exception& ex) {
    std::cerr << "executor_runner error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


// Next Steps

// Configure & build: mkdir -p cmake-out && cd cmake-out && cmake .. && cmake --build .
// Generate and replay: python -m examples.export.export_example --model_name add then ./cmake-out/executor_runner --model_path add.ff
// If everything prints the captured outputs, we can proceed to Milestone 4 (kernels/backends or richer runtime semantics).

