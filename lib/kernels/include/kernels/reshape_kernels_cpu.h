#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Reshape {

void cpu_forward_kernel(DataType data_type,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void cpu_backward_kernel(DataType data_type,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace FlexFlow

#endif
