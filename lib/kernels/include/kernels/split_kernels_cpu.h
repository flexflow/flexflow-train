#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_CPU_H

namespace FlexFlow::Kernels::Split {

void cpu_forward_kernel(float **out_ptrs,
                    float const *in_ptr,
                    int const *out_blk_sizes,
                    int in_blk_size,
                    int num_blks,
                    int numOutputs);

void cpu_backward_kernel(float *in_grad_ptr,
                     float const **out_grad_ptr,
                     int const *out_blk_sizes,
                     int in_blk_size,
                     int num_blks,
                     int numOutputs);

} // namespace FlexFlow

#endif
