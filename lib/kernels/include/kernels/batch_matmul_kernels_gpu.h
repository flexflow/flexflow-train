#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow::Kernels::BatchMatmul {

void gpu_forward_kernel(ffStream_t stream,
                    PerDeviceFFHandle const &handle,
                    float *output_ptr,
                    float const *a_input_ptr,
                    float const *b_input_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int seq_length,
                    int a_seq_length_dim,
                    int b_seq_length_dim);

void gpu_backward_kernel(ffStream_t stream,
                     PerDeviceFFHandle const &handle,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch);


} // namespace FlexFlow

#endif
