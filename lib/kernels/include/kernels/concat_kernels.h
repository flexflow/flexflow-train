#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Concat {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorW const &output,
                    std::vector<GenericTensorAccessorR> const &inputs,
                    ff_dim_t axis);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output_grad,
                     std::vector<GenericTensorAccessorW> const &input_grads,
                     ff_dim_t axis);

} // namespace FlexFlow::Kernels::Concat

#endif
