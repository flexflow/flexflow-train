#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include <optional>

namespace FlexFlow {

void linear_cpu_forward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &projection,
    std::optional<GenericTensorAccessorR> const &bias);

void linear_cpu_backward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &projection,
    GenericTensorAccessorW const &projection_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad);

} // namespace FlexFlow

#endif
