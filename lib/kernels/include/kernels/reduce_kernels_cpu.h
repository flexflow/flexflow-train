#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_CPU_H

namespace FlexFlow::Kernels::Reduce {

void cpu_forward_kernel(float const *input_ptr, float *output_ptr);

void cpu_backward_kernel(float const *output_grad_ptr, float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Reduce

#endif
