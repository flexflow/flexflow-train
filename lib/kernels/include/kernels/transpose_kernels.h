#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"

namespace FlexFlow::Kernels::Transpose {

void forward_kernel(device_stream_t const &stream,
                    TransposeAttrs const &attrs,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     TransposeAttrs const &attrs,
                     GenericTensorAccessorR const &out_grad,
                     GenericTensorAccessorW const &in_grad);

} // namespace FlexFlow::Kernels::Transpose

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
