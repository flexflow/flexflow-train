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

void cpu_backward_kernel(float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *kernel_ptr,
                         float *kernel_grad_ptr,
                         float *bias_grad_ptr,
                         int in_dim,
                         int out_dim,
                         int batch_size);

} // namespace FlexFlow::Kernels::Linear

#endif
