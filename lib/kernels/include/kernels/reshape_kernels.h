#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "utils/required_core.h"

namespace FlexFlow {

namespace Kernels::Reshape {

ReshapePerDeviceState init_kernel(DataType data_type);

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace Kernels::Reshape
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
