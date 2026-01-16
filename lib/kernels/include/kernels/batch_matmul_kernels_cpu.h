#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_CPU_H

#include "kernels/allocation.h"

namespace FlexFlow::Kernels::BatchMatmul {

void cpu_forward_kernel(GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &input_a,
                        GenericTensorAccessorR const &input_b,
                        positive_int seq_length,
                        std::optional<positive_int> a_seq_length_dim,
                        std::optional<positive_int> b_seq_length_dim);

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input_a,
                         GenericTensorAccessorW const &input_a_grad,
                         GenericTensorAccessorR const &input_b,
                         GenericTensorAccessorW const &input_b_grad);

} // namespace FlexFlow::Kernels::BatchMatmul

#endif
