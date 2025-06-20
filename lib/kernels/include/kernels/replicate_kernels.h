#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Replicate {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input,
                     size_t num_replicas);

} // namespace FlexFlow::Kernels::Replicate

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
