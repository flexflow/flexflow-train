#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_GPU_H

#include "kernels/device.h"

namespace FlexFlow::Kernels::Split {

void gpu_forward_kernel(ffStream_t stream,
                        float **out_ptrs,
                        float const *in_ptr,
                        int const *out_blk_sizes,
                        int in_blk_size,
                        int num_blks,
                        int numOutputs);

void gpu_backward_kernel(ffStream_t stream,
                         float *in_grad_ptr,
                         float const **out_grad_ptr,
                         int const *out_blk_sizes,
                         int in_blk_size,
                         int num_blks,
                         int numOutputs);

} // namespace FlexFlow::Kernels::Split

#endif
