#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCTION_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCTION_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Reduction {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        size_t num_replicas);

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Reduction

#endif
