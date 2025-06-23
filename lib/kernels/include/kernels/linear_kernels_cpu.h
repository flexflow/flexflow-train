#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LINEAR_KERNELS_CPU_H

namespace FlexFlow::Kernels::Linear {

void cpu_forward_kernel(float const *input_ptr,
                        float *output_ptr,
                        float const *filter_ptr,
                        float const *bias_ptr,
                        int in_dim,
                        int out_dim,
                        int batch_size);

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
