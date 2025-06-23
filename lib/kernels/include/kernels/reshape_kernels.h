#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Reshape {

void forward_kernel(device_stream_t const &stream,
                    DataType data_type,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     DataType data_type,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Reshape

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
