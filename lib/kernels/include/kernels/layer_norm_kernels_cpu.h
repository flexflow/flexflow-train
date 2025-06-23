#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow::Kernels::LayerNorm {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorW const &gamma,
                        GenericTensorAccessorW const &beta);

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad,
                         GenericTensorAccessorR const &gamma,
                         GenericTensorAccessorW const &gamma_grad,
                         GenericTensorAccessorW const &beta_grad);

} // namespace FlexFlow::Kernels::LayerNorm

#endif
