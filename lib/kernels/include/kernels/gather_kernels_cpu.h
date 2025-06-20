#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GATHER_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GATHER_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Gather {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output);

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad);


} // namespace FlexFlow

#endif
