#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Flat {

void cpu_forward_kernel(GenericTensorAccessorR const &input, float *output_ptr);

void cpu_backward_kernel(GenericTensorAccessorR const &input,
                         float const *output_grad_ptr,
                         float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Flat

#endif
