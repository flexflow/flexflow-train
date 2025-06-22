#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H

namespace FlexFlow::Kernels::Pool2D {

void cpu_forward_kernel(void const *input_ptr,
                    void *output_ptr);

void cpu_backward_kernel(void const *output_ptr,
                     void const *output_grad_ptr,
                     void const *input_ptr,
                     void *input_grad_ptr);


} // namespace FlexFlow

#endif
