#include "kernels/batch_matmul_kernels_cpu.h"

namespace FlexFlow::Kernels::BatchMatmul {

void cpu_forward_kernel(GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &input_a,
                        GenericTensorAccessorR const &input_b,
                        positive_int seq_length,
                        std::optional<positive_int> a_seq_length_dim,
                        std::optional<positive_int> b_seq_length_dim) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input_a,
                         GenericTensorAccessorW const &input_a_grad,
                         GenericTensorAccessorR const &input_b,
                         GenericTensorAccessorW const &input_b_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::BatchMatmul
