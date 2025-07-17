#include "kernels/batch_matmul_kernels_cpu.h"

namespace FlexFlow::Kernels::BatchMatmul {

void cpu_forward_kernel(float *output_ptr,
                        float const *a_input_ptr,
                        float const *b_input_ptr,
                        int m,
                        int n,
                        int k,
                        int batch,
                        int seq_length,
                        int a_seq_length_dim,
                        int b_seq_length_dim) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *o_ptr,
                         float const *o_grad_ptr,
                         float const *a_ptr,
                         float *a_grad_ptr,
                         float const *b_ptr,
                         float *b_grad_ptr,
                         int m,
                         int n,
                         int k,
                         int batch) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::BatchMatmul
