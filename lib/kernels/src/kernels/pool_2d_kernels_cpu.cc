#include "kernels/pool_2d_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow::Kernels::Pool2D {

void cpu_forward_kernel(void const *input_ptr,
                    void *output_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(void const *output_ptr,
                     void const *output_grad_ptr,
                     void const *input_ptr,
                     void *input_grad_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
