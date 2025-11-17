#pragma once

#include <string>
#include <unordered_map>
namespace minixt {
struct Instruction {
  std::unordered_map<std::string, double> numeric;
  std::unordered_map<std::string, std::string> strings;
};
inline double get_numeric_attr(const Instruction &instr, const std::string &key,
                               double default_value) {
  const auto it = instr.numeric.find(key);
  return (it == instr.numeric.end()) ? default_value : it->second;
}
} // namespace minixt