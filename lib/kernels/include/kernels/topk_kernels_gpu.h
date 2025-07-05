#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TOPK_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TOPK_KERNELS_GPU_H

#include "kernels/device.h"

namespace FlexFlow::Kernels::TopK {

void gpu_forward_kernel(ffStream_t stream,
                        float const *input_ptr,
                        float *output_ptr,
                        int *indices_ptr,
                        size_t batch_size,
                        int length,
                        int k,
                        bool sorted);

void gpu_backward_kernel(ffStream_t stream,
                         float const *out_grad_ptr,
                         int const *indices_ptr,
                         float *in_grad_ptr,
                         size_t batch_size,
                         int length,
                         int k);

} // namespace FlexFlow::Kernels::TopK

#endif
