#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include <vector>

namespace FlexFlow {

namespace Kernels::Transpose {

void forward_kernel(cudaStream_t stream,
                    TransposeAttrs const &attrs,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(cudaStream_t stream,
                     TransposeAttrs const &attrs,
                     GenericTensorAccessorR const &out_grad,
                     GenericTensorAccessorW const &in_grad);

} // namespace Kernels::Transpose
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
