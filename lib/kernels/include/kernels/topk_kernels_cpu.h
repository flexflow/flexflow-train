#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TOPK_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TOPK_KERNELS_CPU_H

#include <cstddef>

namespace FlexFlow::Kernels::TopK {

void cpu_forward_kernel(float const *input_ptr,
                        float *output_ptr,
                        int *indices_ptr,
                        size_t batch_size,
                        int length,
                        int k,
                        bool sorted);

void cpu_backward_kernel(float const *out_grad_ptr,
                         int const *indices_ptr,
                         float *in_grad_ptr,
                         size_t batch_size,
                         int length,
                         int k);

} // namespace FlexFlow::Kernels::TopK

#endif
