#include "kernels/linear_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow::Kernels::Linear {

void cpu_forward_kernel(float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *kernel_ptr,
                     float *kernel_grad_ptr,
                     float *bias_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
