#include "kernels/cpu/register.h"
// #include "runtime/core/tensor.h"
// #include "runtime/executor/graph.h"
#include "runtime/kernel/registry.h"
#include <cmath>
#include <stdexcept>

namespace executorch::kernels::cpu {
namespace {
using executorch::runtime::core::Tensor;
using executorch::runtime::executor::Instruction;
using executorch::runtime::kernel::ParameterMap;
using executorch::runtime::kernel::TensorRefVector;

const Tensor &require_parameter(const ParameterMap &parameters,
                                const std::string &name) {
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    throw std::runtime_error("Missing parameter '" + name + "'");
  }
  return it->second;
}

double get_numeric_attr(const Instruction &instruction, const std::string &key,
                        double default_value) {
  const auto it = instruction.attributes.numeric.find(key);
  return (it == instruction.attributes.numeric.end()) ? default_value
                                                      : it->second;
}

Tensor clone_shape(const Tensor &source) {
  Tensor result;
  result.dtype = source.dtype;
  result.shape = source.shape;
  result.values = source.values;
  return result;
}

std::vector<Tensor> add_kernel(const Instruction &instruction,
                               const TensorRefVector &inputs,
                               const ParameterMap &parameters) {
  if (inputs.size() != 2) {
    throw std::runtime_error("aten::add expects exactly two inputs");
  }
  const Tensor &lhs = *inputs[0];
  const Tensor &rhs = *inputs[1];

  if (lhs.shape != rhs.shape) {
    throw std::runtime_error(
        "aten::add requires inputs to share the same shape");
  }
  Tensor result;
  result.dtype = lhs.dtype;
  result.shape = lhs.shape;
  result.values.resize(lhs.values.size());

  double alpha = 1.0;
  const auto alpha_it = instruction.attributes.numeric.find("alpha");
  if (alpha_it != instruction.attributes.numeric.end()) {
    alpha = alpha_it->second;
  }
  for (std::size_t i = 0; i < lhs.values.size(); ++i) {
    result.values[i] = lhs.values[i] + alpha * rhs.values[i];
  }
  return {std::move(result)};
}

std::vector<Tensor> linear_kernel(const Instruction &instruction,
                                  const TensorRefVector &inputs,
                                  const ParameterMap &parameters) {
  if (inputs.size() != 1) {
    throw std::runtime_error(
        "aten::linear expects exactly one activation input");
  }
  const Tensor &activation = *inputs[0];
  const auto weight_name_it = instruction.attributes.strings.find("weight");
  if (weight_name_it == instruction.attributes.strings.end()) {
    throw std::runtime_error("aten::linear missing weight attribute");
  }
  const auto weight_it = parameters.find(weight_name_it->second);
  if (weight_it == parameters.end()) {
    throw std::runtime_error(
        "aten::linear could not locate weight parameter '" +
        weight_name_it->second + "'");
  }
  const Tensor &weight = weight_it->second;

  const Tensor *bias = nullptr;
  const auto bias_name_it = instruction.attributes.strings.find("bias");
  if (bias_name_it != instruction.attributes.strings.end()) {
    const auto bias_it = parameters.find(bias_name_it->second);
    if (bias_it == parameters.end()) {
      throw std::runtime_error(
          "aten::linear could not locate bias parameter '" +
          bias_name_it->second + "'");
    }
    bias = &bias_it->second;
  }

  if (weight.shape.size() != 2) {
    throw std::runtime_error("aten::linear expects rank-2 weight tensor");
  }

  if (activation.shape.empty()) {
    throw std::runtime_error("aten::linear expects activation with >=1 dimension");
  }
  const auto in_features =
      static_cast<std::size_t>(activation.shape.back());
  const auto out_features = static_cast<std::size_t>(weight.shape[0]);
  const auto weight_in_features = static_cast<std::size_t>(weight.shape[1]);

  if (in_features != weight_in_features) {
    throw std::runtime_error(
        "aten::linear shape mismatch between activation and weight");
  }
  if (bias && bias->values.size() != out_features) {
    throw std::runtime_error("aten::linear bias size mismatch");
  }

  const auto total_elements = activation.values.size();
  if (in_features == 0 || total_elements % in_features != 0) {
    throw std::runtime_error("aten::linear activation size not divisible by in_features");
  }
  const auto batch = total_elements / in_features;

  Tensor result;
  result.dtype = activation.dtype;
  result.shape = activation.shape;
  result.shape.back() = static_cast<int64_t>(out_features);
  result.values.resize(batch * out_features, 0.0);

  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t o = 0; o < out_features; ++o) {
      double accumulator = bias ? bias->values[o] : 0.0;
      for (std::size_t i = 0; i < in_features; ++i) {
        const double activation_value =
            activation.values[b * in_features + i];
        const double weight_value = weight.values[o * in_features + i];
        accumulator += activation_value * weight_value;
      }
      result.values[b * out_features + o] = accumulator;
    }
  }

  return {std::move(result)};
}

std::vector<Tensor> relu_kernel(const Instruction &,
                                const TensorRefVector &inputs,
                                const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::relu expects 1 input");
  }
  Tensor out = clone_shape(*inputs[0]);
  for (double &value : out.values) {
    if (value < 0.0) {
      value = 0.0;
    }
  }
  return {std::move(out)};
}

std::vector<Tensor> hardtanh_kernel(const Instruction &instruction,
                                    const TensorRefVector &inputs,
                                    const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::hardtanh expects 1 input");
  }
  const double min_val = get_numeric_attr(instruction, "min_val", 0.0);
  const double max_val = get_numeric_attr(instruction, "max_val", 6.0);
  Tensor out = clone_shape(*inputs[0]);
  for (double &value : out.values) {
    if (value < min_val) {
      value = min_val;
    } else if (value > max_val) {
      value = max_val;
    }
  }
  return {std::move(out)};
}

std::vector<Tensor> dropout_kernel(const Instruction &instruction,
                                   const TensorRefVector &inputs,
                                   const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::dropout expects 1 input");
  }
  const bool training = get_numeric_attr(instruction, "training", 0.0) != 0.0;
  if (training) {
    throw std::runtime_error("aten::dropout only supports eval mode (training False)");
  }
  // Dropout is a no-op during inference, so just forward the tensor.
  return {clone_shape(*inputs[0])};
}

std::vector<Tensor> hardswish_kernel(const Instruction &,
                                     const TensorRefVector &inputs,
                                     const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::hardswish expects 1 input");
  }
  Tensor out = clone_shape(*inputs[0]);
  for (double &value : out.values) {
    double tmp = value + 3.0;
    if (tmp < 0.0) {
      tmp = 0.0;
    } else if (tmp > 6.0) {
      tmp = 6.0;
    }
    value = value * tmp / 6.0;
  }
  return {std::move(out)};
}

std::vector<Tensor> flatten_kernel(const Instruction &instruction,
                                   const TensorRefVector &inputs,
                                   const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::flatten expects 1 input");
  }
  const Tensor &input = *inputs[0];
  const auto rank = static_cast<int64_t>(input.shape.size());
  int64_t start_dim =
      static_cast<int64_t>(get_numeric_attr(instruction, "start_dim", 0));
  int64_t end_dim =
      static_cast<int64_t>(get_numeric_attr(instruction, "end_dim", -1));
  if (start_dim < 0) {
    start_dim += rank;
  }
  if (end_dim < 0) {
    end_dim += rank;
  }
  if (start_dim < 0 || end_dim >= rank || start_dim > end_dim) {
    throw std::runtime_error("aten::flatten received invalid dims");
  }

  Tensor out;
  out.dtype = input.dtype;
  out.shape.reserve(rank - (end_dim - start_dim));
  for (int64_t i = 0; i < start_dim; ++i) {
    out.shape.push_back(input.shape[static_cast<std::size_t>(i)]);
  }
  int64_t collapsed = 1;
  for (int64_t i = start_dim; i <= end_dim; ++i) {
    collapsed *= input.shape[static_cast<std::size_t>(i)];
  }
  out.shape.push_back(collapsed);
  for (int64_t i = end_dim + 1; i < rank; ++i) {
    out.shape.push_back(input.shape[static_cast<std::size_t>(i)]);
  }
  out.values = input.values;
  return {std::move(out)};
}

std::vector<Tensor> adaptive_avg_pool2d_kernel(const Instruction &instruction,
                                               const TensorRefVector &inputs,
                                               const ParameterMap &) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::adaptive_avg_pool2d expects 1 input");
  }
  const Tensor &input = *inputs[0];
  if (input.shape.size() != 4) {
    throw std::runtime_error("aten::adaptive_avg_pool2d expects NCHW input");
  }

  const auto batch = static_cast<std::size_t>(input.shape[0]);
  const auto channels = static_cast<std::size_t>(input.shape[1]);
  const auto in_h = static_cast<std::size_t>(input.shape[2]);
  const auto in_w = static_cast<std::size_t>(input.shape[3]);

  const auto out_h =
      static_cast<std::size_t>(get_numeric_attr(instruction, "output_h", 1));
  const auto out_w =
      static_cast<std::size_t>(get_numeric_attr(instruction, "output_w", 1));

  Tensor out;
  out.dtype = input.dtype;
  out.shape = {
      static_cast<int64_t>(batch),
      static_cast<int64_t>(channels),
      static_cast<int64_t>(out_h),
      static_cast<int64_t>(out_w),
  };
  out.values.resize(batch * channels * out_h * out_w);

  auto kernel_size = [](std::size_t in_dim, std::size_t out_dim,
                        std::size_t idx) {
    const std::size_t start =
        std::floor(static_cast<double>(idx * in_dim) / out_dim);
    const std::size_t end =
        std::ceil(static_cast<double>((idx + 1) * in_dim) / out_dim);
    return std::pair<std::size_t, std::size_t>{start, end};
  };

  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t c = 0; c < channels; ++c) {
      for (std::size_t oh = 0; oh < out_h; ++oh) {
        const auto [h_start, h_end] = kernel_size(in_h, out_h, oh);
        for (std::size_t ow = 0; ow < out_w; ++ow) {
          const auto [w_start, w_end] = kernel_size(in_w, out_w, ow);
          double acc = 0.0;
          std::size_t count = 0;
          for (std::size_t ih = h_start; ih < h_end; ++ih) {
            for (std::size_t iw = w_start; iw < w_end; ++iw) {
              const std::size_t idx =
                  ((b * channels + c) * in_h + ih) * in_w + iw;
              acc += input.values[idx];
              ++count;
            }
          }
          const std::size_t out_idx =
              ((b * channels + c) * out_h + oh) * out_w + ow;
          out.values[out_idx] = acc / static_cast<double>(count);
        }
      }
    }
  }

  return {std::move(out)};
}

std::vector<Tensor> batch_norm_kernel(const Instruction &instruction,
                                      const TensorRefVector &inputs,
                                      const ParameterMap &parameters) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::batch_norm expects 1 input");
  }
  const Tensor &input = *inputs[0];
  if (input.shape.size() != 4) {
    throw std::runtime_error(
        "aten::batch_norm currently supports NCHW tensors");
  }

  const auto &strings = instruction.attributes.strings;
  const Tensor &weight = require_parameter(parameters, strings.at("weight"));
  const Tensor &bias = require_parameter(parameters, strings.at("bias"));
  const Tensor &running_mean =
      require_parameter(parameters, strings.at("running_mean"));
  const Tensor &running_var =
      require_parameter(parameters, strings.at("running_var"));

  const auto channels = static_cast<std::size_t>(input.shape[1]);
  const double eps = get_numeric_attr(instruction, "eps", 1e-5);

  Tensor out = clone_shape(input);
  const auto spatial =
      static_cast<std::size_t>(input.shape[2] * input.shape[3]);

  for (std::size_t b = 0; b < static_cast<std::size_t>(input.shape[0]); ++b) {
    for (std::size_t c = 0; c < channels; ++c) {
      const double inv_std = 1.0 / std::sqrt(running_var.values[c] + eps);
      const double w = weight.values[c];
      const double b_val = bias.values[c];
      for (std::size_t idx = 0; idx < spatial; ++idx) {
        const std::size_t offset = ((b * channels + c) * spatial) + idx;
        const double normalized =
            (input.values[offset] - running_mean.values[c]) * inv_std;
        out.values[offset] = normalized * w + b_val;
      }
    }
  }

  return {std::move(out)};
}

std::vector<Tensor> conv2d_kernel(const Instruction &instruction,
                                  const TensorRefVector &inputs,
                                  const ParameterMap &parameters) {
  if (inputs.size() != 1) {
    throw std::runtime_error("aten::conv2d expects 1 activation input");
  }
  const Tensor &input = *inputs[0];
  if (input.shape.size() != 4) {
    throw std::runtime_error("aten::conv2d expects NCHW activation");
  }

  const auto &strings = instruction.attributes.strings;
  const Tensor &weight = require_parameter(parameters, strings.at("weight"));
  const Tensor *bias = nullptr;
  const auto bias_it = strings.find("bias");
  if (bias_it != strings.end()) {
    bias = &require_parameter(parameters, bias_it->second);
  }

  const auto stride_h =
      static_cast<int64_t>(get_numeric_attr(instruction, "stride_h", 1));
  const auto stride_w =
      static_cast<int64_t>(get_numeric_attr(instruction, "stride_w", 1));
  const auto pad_h =
      static_cast<int64_t>(get_numeric_attr(instruction, "padding_h", 0));
  const auto pad_w =
      static_cast<int64_t>(get_numeric_attr(instruction, "padding_w", 0));
  const auto dil_h =
      static_cast<int64_t>(get_numeric_attr(instruction, "dilation_h", 1));
  const auto dil_w =
      static_cast<int64_t>(get_numeric_attr(instruction, "dilation_w", 1));
  const auto groups =
      static_cast<int64_t>(get_numeric_attr(instruction, "groups", 1));

  const auto batch = static_cast<int64_t>(input.shape[0]);
  const auto in_channels = static_cast<int64_t>(input.shape[1]);
  const auto in_h = static_cast<int64_t>(input.shape[2]);
  const auto in_w = static_cast<int64_t>(input.shape[3]);

  const auto out_channels = static_cast<int64_t>(weight.shape[0]);
  const auto kernel_h = static_cast<int64_t>(weight.shape[2]);
  const auto kernel_w = static_cast<int64_t>(weight.shape[3]);

  const auto out_h =
      (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
  const auto out_w =
      (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

  Tensor out;
  out.dtype = input.dtype;
  out.shape = {batch, out_channels, out_h, out_w};
  out.values.assign(
      static_cast<std::size_t>(batch * out_channels * out_h * out_w), 0.0);

  const int64_t channels_per_group = in_channels / groups;
  const int64_t kernels_per_group = out_channels / groups;

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t g = 0; g < groups; ++g) {
      for (int64_t oc = 0; oc < kernels_per_group; ++oc) {
        const int64_t out_channel = g * kernels_per_group + oc;
        for (int64_t oh = 0; oh < out_h; ++oh) {
          for (int64_t ow = 0; ow < out_w; ++ow) {
            double acc =
                bias ? bias->values[static_cast<std::size_t>(out_channel)]
                     : 0.0;
            for (int64_t ic = 0; ic < channels_per_group; ++ic) {
              const int64_t in_channel = g * channels_per_group + ic;
              for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                  const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                  const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                  if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                    continue;
                  }
                  const auto input_index =
                      ((b * in_channels + in_channel) * in_h + ih) * in_w + iw;
                  const auto weight_index =
                      (((out_channel)*channels_per_group + ic) * kernel_h +
                       kh) *
                          kernel_w +
                      kw;
                  acc += input.values[static_cast<std::size_t>(input_index)] *
                         weight.values[static_cast<std::size_t>(weight_index)];
                }
              }
            }
            const auto out_index =
                ((b * out_channels + out_channel) * out_h + oh) * out_w + ow;
            out.values[static_cast<std::size_t>(out_index)] = acc;
          }
        }
      }
    }
  }

  return {std::move(out)};
}

} // namespace
void register_kernels() {
  auto &registry = executorch::runtime::kernel::KernelRegistry::instance();
  registry.register_kernel("aten::add", add_kernel);
  registry.register_kernel("aten::linear", linear_kernel);
  registry.register_kernel("aten::relu", relu_kernel);
  registry.register_kernel("aten::hardtanh", hardtanh_kernel);
  registry.register_kernel("aten::dropout", dropout_kernel);
  registry.register_kernel("aten::hardswish", hardswish_kernel);
  registry.register_kernel("aten::flatten", flatten_kernel);
  registry.register_kernel("aten::adaptive_avg_pool2d",
                           adaptive_avg_pool2d_kernel);
  registry.register_kernel("aten::batch_norm", batch_norm_kernel);
  registry.register_kernel("aten::conv2d", conv2d_kernel);
}
} // namespace executorch::kernels::cpu
