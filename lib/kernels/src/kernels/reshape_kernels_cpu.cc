#include "kernels/reshape_kernels_cpu.h"

namespace FlexFlow::Kernels::Reshape {

void cpu_forward_kernel(DataType data_type,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(DataType data_type,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
