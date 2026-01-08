#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow::Kernels::BatchMatmul {

void gpu_forward_kernel(ffStream_t stream,
                        PerDeviceFFHandle const &handle,
                        float *output_ptr,
                        float const *input_a_ptr,
                        float const *input_b_ptr,
                        int m,
                        int n,
                        int k,
                        int batch,
                        int seq_length,
                        int a_seq_length_dim,
                        int b_seq_length_dim);

void gpu_backward_kernel(ffStream_t stream,
                         PerDeviceFFHandle const &handle,
                         float const *output_ptr,
                         float const *output_grad_ptr,
                         float const *input_a_ptr,
                         float *input_a_grad_ptr,
                         float const *input_b_ptr,
                         float *input_b_grad_ptr,
                         int m,
                         int n,
                         int k,
                         int batch);

} // namespace FlexFlow::Kernels::BatchMatmul

#endif
