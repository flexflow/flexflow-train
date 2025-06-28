#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_CPU_H

#include <cstddef>

namespace FlexFlow::Kernels::Softmax {

void cpu_forward_kernel(float const *input_ptr, float *output_ptr);

void cpu_backward_kernel(float const *output_grad_ptr,
                         float *input_grad_ptr,
                         size_t num_elements);

} // namespace FlexFlow::Kernels::Softmax

#endif
