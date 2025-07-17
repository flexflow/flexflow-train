#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Cast {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow::Kernels::Cast

#endif
