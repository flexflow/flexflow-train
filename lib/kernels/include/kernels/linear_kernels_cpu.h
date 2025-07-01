#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H

#include "kernels/accessor.h"
#include <optional>

namespace FlexFlow::Kernels::Linear {

void cpu_forward_kernel(
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW const &output_accessor,
                    GenericTensorAccessorR const &filter_accessor,
                    std::optional<GenericTensorAccessorR> const &bias_accessor);

void cpu_backward_kernel(
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad,
                         GenericTensorAccessorR const &kernel,
                         GenericTensorAccessorW const &kernel_grad,
                         std::optional<GenericTensorAccessorW> const &bias_grad);

} // namespace FlexFlow::Kernels::Linear

#endif
