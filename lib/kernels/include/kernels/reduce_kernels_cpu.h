#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/reduce_attrs.dtg.h"

namespace FlexFlow {

void reduce_cpu_forward_kernel(ReduceAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output);

void reduce_cpu_backward_kernel(ReduceAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
