#include "runtime/kernel/registry.h"

#include <utility>

namespace executorch::runtime::kernel {

KernelRegistry &KernelRegistry::instance() {
  static KernelRegistry registry;
  return registry;
}

void KernelRegistry::register_kernel(const std::string &op, KernelFn fn) {
  kernels_[op] = std::move(fn);
}

const KernelFn *KernelRegistry::lookup(const std::string &op) const {
  const auto it = kernels_.find(op);
  if (it == kernels_.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace executorch::runtime::kernel
