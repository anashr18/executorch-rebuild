#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace executorch::runtime::executor {

struct InstructionAttributes {
  std::unordered_map<std::string, double> numeric;
  std::unordered_map<std::string, std::string> strings;
};

struct Instruction {
  std::string name;
  std::string op;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  InstructionAttributes attributes;
};

struct ProgramGraph {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<Instruction> instructions;
};

} // namespace executorch::runtime::executor
