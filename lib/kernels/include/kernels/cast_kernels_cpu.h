#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Cast {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output);

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Cast

#endif
