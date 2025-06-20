#include "kernels/reduction_kernels_cpu.h"

namespace FlexFlow::Kernels::Reduction {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    size_t num_replicas) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
