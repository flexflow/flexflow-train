#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"

namespace FlexFlow::Kernels::BatchMatmul {

void forward_kernel(device_stream_t const &stream,
                    device_handle_t const &handle,
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

void backward_kernel(device_stream_t const &stream,
                     device_handle_t const &handle,
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

} // namespace FlexFlow::Kernels::BatchMatmul

#endif
