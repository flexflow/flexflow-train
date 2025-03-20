#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "kernels/device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Cast {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Cast

#endif
