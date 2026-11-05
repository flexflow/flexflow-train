#include "kernels/reduce_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void reduce_cpu_forward_kernel(ReduceAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void reduce_cpu_backward_kernel(ReduceAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
