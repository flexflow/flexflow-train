#include "kernels/partition_kernels_cpu.h"

namespace FlexFlow::Kernels::Repartition {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Repartition
