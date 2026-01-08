#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow::Kernels::BatchMatmul {

void forward_kernel(device_stream_t const &stream,
                    device_handle_t const &handle,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &input_a,
                    GenericTensorAccessorR const &input_b,
                    positive_int seq_length,
                    std::optional<positive_int> a_seq_length_dim,
                    std::optional<positive_int> b_seq_length_dim);

void backward_kernel(device_stream_t const &stream,
                     device_handle_t const &handle,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input_a,
                     GenericTensorAccessorW const &input_a_grad,
                     GenericTensorAccessorR const &input_b,
                     GenericTensorAccessorW const &input_b_grad);

} // namespace FlexFlow::Kernels::BatchMatmul

#endif
