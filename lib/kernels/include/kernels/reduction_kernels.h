#ifndef _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow::Kernels::Reduction {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    size_t num_replicas);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Reduction

#endif // _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
