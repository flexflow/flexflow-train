#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REPLICATE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REPLICATE_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow::Kernels::Replicate {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output);

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input,
                         size_t num_replicas);

} // namespace FlexFlow::Kernels::Replicate

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H
