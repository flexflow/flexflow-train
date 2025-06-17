#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_CPU_H

#include "op-attrs/activation.dtg.h"
#include <optional>

namespace FlexFlow::Kernels::Conv2D {

void cpu_forward_kernel(float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    std::optional<Activation> const &activation);

void cpu_backward_kernel(float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr,
                     std::optional<Activation> const &activation);


} // namespace FlexFlow

#endif
